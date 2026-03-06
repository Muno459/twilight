// Twilight MCRT - WGSL compute shaders for WebGPU (v2)
//
// Four compute kernels for GPU-accelerated twilight radiative transfer:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_scatter            - LOS + secondary MC chains (reparallelized)
//   4. garstang_zenith           - Light pollution skyglow
//
// All four kernels are in one module with separate @compute entry points.
// Buffer layout matches crates/twilight-gpu/src/buffers.rs (v2) exactly.
// All physics ported from twilight-core (f64) to f32 GPU precision.
// Uses xorshift32 RNG (WGSL has no native u64).
//
// Key changes from v1:
//   - Binary search O(log N) shell lookup
//   - Shell-by-shell shadow ray with Snell's law refraction
//   - Radial 2m boundary nudge (not along ray direction)
//   - Kahan compensated summation for optical depth and radiance
//   - Hybrid kernel: 1 workgroup per wavelength with shared memory reduction

enable subgroups;

// ============================================================================
// Buffer bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read> params_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_buf: array<f32>;

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265358979323846;
const EARTH_RADIUS_M: f32 = 6371000.0;
const TOA_ALTITUDE_M: f32 = 100000.0;

const MAX_WAVELENGTHS: u32 = 64u;
const MAX_LOS_STEPS: u32 = 200u;
const MAX_SCATTERS: u32 = 100u;
const HYBRID_LOS_STEPS: u32 = 200u;
const HYBRID_MAX_BOUNCES: u32 = 50u;
const HYBRID_THREADGROUP_SIZE: u32 = 256u;

// Atmosphere buffer offsets (must match buffers.rs atm_offsets)
const ATM_NUM_SHELLS: u32 = 2u;
const ATM_NUM_WAVELENGTHS: u32 = 3u;
const ATM_SHELLS_START: u32 = 4u;
const ATM_SHELL_STRIDE: u32 = 4u;
const ATM_OPTICS_START: u32 = 260u;   // 4 + 4*64
const ATM_OPTICS_STRIDE: u32 = 4u;
const ATM_ALBEDO_START: u32 = 16708u; // 16644 + 64
const ATM_REFRACTIVE_INDEX_START: u32 = 16772u; // 16708 + 64 (v2)

// Garstang constants
const H_RAYLEIGH: f32 = 8500.0;
const H_AEROSOL: f32 = 1500.0;
const TAU_RAYLEIGH_550: f32 = 0.0962;

// Boundary nudge distance (meters)
const BOUNDARY_NUDGE_M: f32 = 2.0;

// ============================================================================
// Structs
// ============================================================================

struct ShellGeom {
    r_inner: f32,
    r_outer: f32,
    altitude_mid: f32,
    thickness: f32,
};

struct ShellOptics {
    extinction: f32,
    ssa: f32,
    asymmetry: f32,
    rayleigh_fraction: f32,
};

struct RaySphereHit {
    t_near: f32,
    t_far: f32,
    hit: bool,
};

struct ShellBoundary {
    dist: f32,
    is_outward: bool,
    found: bool,
};

// Kahan accumulator: x=sum, y=compensation
struct KahanAccum {
    sum: f32,
    comp: f32,
};

// ============================================================================
// Kahan summation
// ============================================================================

fn kahan_init() -> KahanAccum { return KahanAccum(0.0, 0.0); }

fn kahan_add(accum: KahanAccum, value: f32) -> KahanAccum {
    let y = value - accum.comp;
    let t = accum.sum + y;
    return KahanAccum(t, (t - accum.sum) - y);
}

fn kahan_result(accum: KahanAccum) -> f32 { return accum.sum + accum.comp; }

// ============================================================================
// Buffer accessor helpers
// ============================================================================

fn atm_num_shells_val() -> u32 {
    return u32(input_buf[ATM_NUM_SHELLS]);
}

fn atm_num_wavelengths_val() -> u32 {
    return u32(input_buf[ATM_NUM_WAVELENGTHS]);
}

fn read_shell(shell_idx: u32) -> ShellGeom {
    let base = ATM_SHELLS_START + shell_idx * ATM_SHELL_STRIDE;
    return ShellGeom(
        input_buf[base + 0u],
        input_buf[base + 1u],
        input_buf[base + 2u],
        input_buf[base + 3u],
    );
}

fn read_optics(shell_idx: u32, wl_idx: u32) -> ShellOptics {
    let idx = shell_idx * MAX_WAVELENGTHS + wl_idx;
    let base = ATM_OPTICS_START + idx * ATM_OPTICS_STRIDE;
    return ShellOptics(
        input_buf[base + 0u],
        input_buf[base + 1u],
        input_buf[base + 2u],
        input_buf[base + 3u],
    );
}

fn read_albedo(wl_idx: u32) -> f32 {
    return input_buf[ATM_ALBEDO_START + wl_idx];
}

fn read_refractive_index(shell_idx: u32) -> f32 {
    return input_buf[ATM_REFRACTIVE_INDEX_START + shell_idx];
}

// Dispatch params: 4 x vec4 (16 floats)
fn read_observer() -> vec3f {
    return vec3f(params_buf[0u], params_buf[1u], params_buf[2u]);
}

fn read_view_dir() -> vec3f {
    return vec3f(params_buf[4u], params_buf[5u], params_buf[6u]);
}

fn read_sun_dir() -> vec3f {
    return vec3f(params_buf[8u], params_buf[9u], params_buf[10u]);
}

fn read_photons_per_wl() -> u32 {
    return bitcast<u32>(params_buf[12u]);
}

fn read_secondary_rays_val() -> u32 {
    return bitcast<u32>(params_buf[13u]);
}

fn read_rng_seed_lo() -> u32 {
    return bitcast<u32>(params_buf[14u]);
}

fn read_rng_seed_hi() -> u32 {
    return bitcast<u32>(params_buf[15u]);
}

// ============================================================================
// xorshift32 RNG (WGSL has no native u64)
// ============================================================================

fn xorshift_f32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return f32(x >> 8u) * (1.0 / f32(1u << 24u));
}

// ============================================================================
// Ray-sphere intersection
// ============================================================================

fn ray_sphere_intersect(origin: vec3f, dir: vec3f, radius: f32) -> RaySphereHit {
    let a = dot(dir, dir);
    let b_half = dot(origin, dir);
    let r_pos = length(origin);
    let c = (r_pos - radius) * (r_pos + radius);
    let disc = b_half * b_half - a * c;

    if disc < 0.0 {
        return RaySphereHit(0.0, 0.0, false);
    }

    let sqrt_disc = sqrt(disc);
    let inv_a = 1.0 / a;
    let t_near = (-b_half - sqrt_disc) * inv_a;
    let t_far = (-b_half + sqrt_disc) * inv_a;
    return RaySphereHit(t_near, t_far, true);
}

// ============================================================================
// Shell index lookup -- O(log N) binary search
// ============================================================================

fn shell_index_binary(r: f32) -> i32 {
    let ns = atm_num_shells_val();
    if ns == 0u { return -1; }

    let r_inner_first = input_buf[ATM_SHELLS_START];
    let r_outer_last = input_buf[ATM_SHELLS_START + (ns - 1u) * ATM_SHELL_STRIDE + 1u];
    if r < r_inner_first || r >= r_outer_last { return -1; }

    var lo = 0u;
    var hi = ns;
    while lo < hi {
        let mid = lo + (hi - lo) / 2u;
        let r_inner_mid = input_buf[ATM_SHELLS_START + mid * ATM_SHELL_STRIDE];
        if r_inner_mid <= r {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if lo == 0u { return -1; }
    return i32(lo - 1u);
}

// ============================================================================
// Phase functions
// ============================================================================

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.75 * (1.0 + cos_theta * cos_theta);
}

fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (denom * sqrt(denom));
}

fn mixed_phase(cos_theta: f32, op: ShellOptics) -> f32 {
    if op.rayleigh_fraction > 0.99 {
        return rayleigh_phase(cos_theta);
    }
    return op.rayleigh_fraction * rayleigh_phase(cos_theta)
         + (1.0 - op.rayleigh_fraction) * henyey_greenstein_phase(cos_theta, op.asymmetry);
}

// ============================================================================
// Next shell boundary
// ============================================================================

fn next_shell_boundary(pos: vec3f, dir: vec3f, r_inner: f32, r_outer: f32) -> ShellBoundary {
    var result = ShellBoundary(1e30, true, false);
    let EPS = 1e-5f;

    let outer = ray_sphere_intersect(pos, dir, r_outer);
    if outer.hit {
        if outer.t_near > EPS {
            let inner = ray_sphere_intersect(pos, dir, r_inner);
            if inner.hit && inner.t_near > EPS && inner.t_near < outer.t_near {
                result.dist = inner.t_near;
                result.is_outward = false;
                result.found = true;
                return result;
            }
            result.dist = outer.t_near;
            result.is_outward = true;
            result.found = true;
            return result;
        }
        if outer.t_far > EPS {
            let inner = ray_sphere_intersect(pos, dir, r_inner);
            if inner.hit && inner.t_near > EPS && inner.t_near < outer.t_far {
                result.dist = inner.t_near;
                result.is_outward = false;
                result.found = true;
                return result;
            }
            result.dist = outer.t_far;
            result.is_outward = true;
            result.found = true;
            return result;
        }
    }

    let inner = ray_sphere_intersect(pos, dir, r_inner);
    if inner.hit && inner.t_near > EPS {
        result.dist = inner.t_near;
        result.is_outward = false;
        result.found = true;
    }
    return result;
}

// ============================================================================
// Snell's law refraction at spherical shell boundary
// ============================================================================

fn refract_at_boundary(dir: vec3f, boundary_pos: vec3f, n_from: f32, n_to: f32) -> vec3f {
    if abs(n_from - n_to) < 1e-7 { return dir; }

    let outward = normalize(boundary_pos);
    let cos_dir_normal = dot(dir, outward);
    var normal = outward;
    if cos_dir_normal >= 0.0 { normal = -outward; }

    let cos_i = -dot(dir, normal);
    let eta = n_from / n_to;
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

    if k < 0.0 {
        return normalize(dir + normal * (2.0 * cos_i));
    }

    let cos_t = sqrt(k);
    let factor = eta * cos_i - cos_t;
    return normalize(dir * eta + normal * factor);
}

// ============================================================================
// Radial boundary nudge
// ============================================================================

fn radial_nudge(boundary_pos: vec3f, is_outward: bool) -> vec3f {
    let bp_r = length(boundary_pos);
    var radial_dir = vec3f(1.0, 0.0, 0.0);
    if bp_r > 1e-10 { radial_dir = boundary_pos / bp_r; }
    var nudge_sign = 1.0f;
    if !is_outward { nudge_sign = -1.0; }
    return boundary_pos + radial_dir * (nudge_sign * BOUNDARY_NUDGE_M);
}

// ============================================================================
// Shadow ray transmittance -- shell-by-shell with refraction + Kahan
// ============================================================================

fn shadow_ray_transmittance(start_pos: vec3f, sun_dir: vec3f, wl_idx: u32) -> f32 {
    let ns = atm_num_shells_val();
    let surface_radius = input_buf[ATM_SHELLS_START]; // r_inner of shell 0

    var pos = start_pos;
    var dir = sun_dir;
    var tau = kahan_init();

    for (var iter = 0u; iter < 200u; iter++) {
        let r = length(pos);

        let sidx = shell_index_binary(r);
        if sidx < 0 { break; }

        let us = u32(sidx);
        let r_inner = input_buf[ATM_SHELLS_START + us * ATM_SHELL_STRIDE];
        let r_outer = input_buf[ATM_SHELLS_START + us * ATM_SHELL_STRIDE + 1u];

        let optics_idx = us * MAX_WAVELENGTHS + wl_idx;
        let extinction = input_buf[ATM_OPTICS_START + optics_idx * ATM_OPTICS_STRIDE];

        let bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if !bnd.found { break; }

        tau = kahan_add(tau, extinction * bnd.dist);

        let boundary_pos_raw = pos + dir * bnd.dist;
        // Snap to exact boundary radius to prevent cumulative f32 position drift
        var target_r = r_outer;
        if !bnd.is_outward { target_r = r_inner; }
        let bp_len = length(boundary_pos_raw);
        var boundary_pos = boundary_pos_raw;
        if bp_len > 0.0f {
            boundary_pos = boundary_pos_raw * (target_r / bp_len);
        }
        let n_from = read_refractive_index(us);
        var next_shell_idx = us + 1u;
        if !bnd.is_outward { next_shell_idx = us - 1u; }
        var n_to = 1.0f;
        if next_shell_idx < ns { n_to = read_refractive_index(next_shell_idx); }

        dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
        pos = radial_nudge(boundary_pos, bnd.is_outward);

        if !bnd.is_outward && length(pos) <= surface_radius + 1.0 {
            return 0.0;
        }

        if kahan_result(tau) > 50.0 { return 0.0; }
    }

    return exp(-kahan_result(tau));
}

// ============================================================================
// Sampling functions
// ============================================================================

fn sample_rayleigh_analytic(xi: f32) -> f32 {
    let q = 8.0 * xi - 4.0;
    let disc = q * q * 0.25 + 1.0;
    let sqrt_disc = sqrt(disc);
    let a_val = -q * 0.5 + sqrt_disc;
    let b_val = -q * 0.5 - sqrt_disc;
    let u = select(-pow(-a_val, 1.0 / 3.0), pow(a_val, 1.0 / 3.0), a_val >= 0.0);
    let v = select(-pow(-b_val, 1.0 / 3.0), pow(b_val, 1.0 / 3.0), b_val >= 0.0);
    let mu = u + v;
    return clamp(mu, -1.0, 1.0);
}

fn sample_henyey_greenstein(xi: f32, g: f32) -> f32 {
    if abs(g) < 1e-6 {
        return 2.0 * xi - 1.0;
    }
    let g2 = g * g;
    let s = (1.0 - g2) / (1.0 - g + 2.0 * g * xi);
    let mu = (1.0 + g2 - s * s) / (2.0 * g);
    return clamp(mu, -1.0, 1.0);
}

fn scatter_direction(dir: vec3f, cos_theta: f32, phi: f32) -> vec3f {
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let cos_phi = cos(phi);
    let sin_phi = sin(phi);

    let w = dir;
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(w.z) < 0.9);
    let u_vec = normalize(cross(w, up));
    let v_vec = cross(w, u_vec);

    let new_dir = sin_theta * cos_phi * u_vec
                + sin_theta * sin_phi * v_vec
                + cos_theta * w;
    return normalize(new_dir);
}

fn sample_hemisphere(normal: vec3f, rng: ptr<function, u32>) -> vec3f {
    let xi1 = xorshift_f32(rng);
    let xi2 = xorshift_f32(rng);
    let cos_theta = sqrt(xi1);
    let phi = 2.0 * PI * xi2;
    return scatter_direction(normal, cos_theta, phi);
}

// ============================================================================
// Kernel 1: single_scatter_spectrum
// ============================================================================

@compute @workgroup_size(256)
fn single_scatter_spectrum(@builtin(global_invocation_id) gid: vec3u) {
    let tid = gid.x;
    let num_wl = atm_num_wavelengths_val();
    if tid >= num_wl { return; }

    let wl_idx = tid;
    let observer_pos = read_observer();
    let view_dir = read_view_dir();
    let sun_dir = read_sun_dir();

    let toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    let surface_radius = EARTH_RADIUS_M;

    let toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    if !toa_hit.hit || toa_hit.t_far <= 0.0 {
        output_buf[tid] = 0.0;
        return;
    }
    let los_max = toa_hit.t_far;

    let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    let hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3 && ground_hit.t_near < los_max;
    var los_end = los_max;
    if hits_ground { los_end = ground_hit.t_near; }

    if los_end <= 0.0 {
        output_buf[tid] = 0.0;
        return;
    }

    let num_steps = min(MAX_LOS_STEPS, u32(los_end / 500.0) + 20u);
    let ds = los_end / f32(num_steps);

    var radiance = kahan_init();
    var tau_obs = kahan_init();
    let cos_theta = dot(sun_dir, -view_dir);

    for (var step = 0u; step < num_steps; step++) {
        let s = (f32(step) + 0.5) * ds;
        let scatter_pos = observer_pos + view_dir * s;
        let r = length(scatter_pos);

        if r > toa_radius || r < surface_radius { continue; }

        let sidx = shell_index_binary(r);
        if sidx < 0 { continue; }

        let op = read_optics(u32(sidx), wl_idx);
        let beta_scat = op.extinction * op.ssa;

        if beta_scat < 1e-30 {
            tau_obs = kahan_add(tau_obs, op.extinction * ds);
            continue;
        }

        // exp(-(A+B)) = exp(-A)*exp(-B) avoids f32 precision loss when
        // adding a small half-step to a large accumulated tau.
        let t_obs = exp(-kahan_result(tau_obs)) * exp(-op.extinction * ds * 0.5);
        if t_obs < 1e-30 { break; }

        let t_sun = shadow_ray_transmittance(scatter_pos, sun_dir, wl_idx);
        if t_sun < 1e-30 {
            tau_obs = kahan_add(tau_obs, op.extinction * ds);
            continue;
        }

        let phase = mixed_phase(cos_theta, op);
        let di = beta_scat * phase / (4.0 * PI) * t_sun * t_obs * ds;
        radiance = kahan_add(radiance, di);
        tau_obs = kahan_add(tau_obs, op.extinction * ds);
    }

    // Ground reflection
    if hits_ground {
        let albedo = read_albedo(wl_idx);
        if albedo > 1e-10 {
            let ground_pos = observer_pos + view_dir * los_end;
            let ground_normal = normalize(ground_pos);
            let cos_sun_incidence = dot(sun_dir, ground_normal);

            if cos_sun_incidence > 0.0 {
                let t_sun_ground = shadow_ray_transmittance(ground_pos, sun_dir, wl_idx);
                let t_obs_ground = exp(-kahan_result(tau_obs));
                radiance = kahan_add(radiance,
                    albedo / PI * cos_sun_incidence * t_sun_ground * t_obs_ground);
            }
        }
    }

    output_buf[tid] = kahan_result(radiance);
}

// ============================================================================
// Kernel 2: mcrt_trace_photon
// ============================================================================

@compute @workgroup_size(256)
fn mcrt_trace_photon(@builtin(global_invocation_id) gid: vec3u) {
    let tid = gid.x;
    let num_wl = atm_num_wavelengths_val();
    let photons_per_wl = read_photons_per_wl();
    let total = num_wl * photons_per_wl;
    if tid >= total { return; }

    let wl_idx = tid / photons_per_wl;
    let photon_idx = tid % photons_per_wl;

    let observer_pos = read_observer();
    let view_dir = read_view_dir();
    let sun_dir = read_sun_dir();

    let base_seed = read_rng_seed_lo();
    var rng = base_seed + wl_idx;
    rng = rng * 1664525u + photon_idx;
    rng = rng * 1013904223u + 1u;

    let surface_radius = EARTH_RADIUS_M;

    var pos = observer_pos;
    var dir = view_dir;
    var weight = 1.0f;
    var result_weight = kahan_init();

    for (var bounce = 0u; bounce < MAX_SCATTERS; bounce++) {
        let r = length(pos);
        let sidx = shell_index_binary(r);
        if sidx < 0 { break; }

        let us = u32(sidx);
        let sh = read_shell(us);
        let op = read_optics(us, wl_idx);

        if op.extinction < 1e-20 {
            let bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
            if !bnd.found { break; }
            pos = radial_nudge(pos + dir * bnd.dist, bnd.is_outward);
            continue;
        }

        let xi = xorshift_f32(&rng);
        let free_path = -log(1.0 - xi + 1e-30) / op.extinction;

        let bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
        if !bnd.found { break; }

        if free_path >= bnd.dist {
            let boundary_pos = pos + dir * bnd.dist;

            if !bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M {
                let albedo = read_albedo(wl_idx);
                weight *= albedo;
                let normal = normalize(boundary_pos);
                dir = sample_hemisphere(normal, &rng);
                pos = radial_nudge(boundary_pos, true);
                continue;
            }

            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        pos = pos + dir * free_path;

        let t_sun = shadow_ray_transmittance(pos, sun_dir, wl_idx);
        if t_sun > 1e-30 {
            let cos_angle = dot(sun_dir, -dir);
            let phase = mixed_phase(cos_angle, op);
            result_weight = kahan_add(result_weight, weight * t_sun * phase / (4.0 * PI));
        }

        weight *= op.ssa;

        if weight < 0.01 {
            let xi_rr = xorshift_f32(&rng);
            if xi_rr > 0.1 { break; }
            weight /= 0.1;
        }

        var cos_theta_s: f32;
        if xorshift_f32(&rng) < op.rayleigh_fraction {
            cos_theta_s = sample_rayleigh_analytic(xorshift_f32(&rng));
        } else {
            cos_theta_s = sample_henyey_greenstein(xorshift_f32(&rng), op.asymmetry);
        }
        let phi = 2.0 * PI * xorshift_f32(&rng);
        dir = scatter_direction(dir, cos_theta_s, phi);
    }

    output_buf[tid] = kahan_result(result_weight);
}

// ============================================================================
// Secondary chain tracer (used by hybrid_scatter)
// ============================================================================

fn trace_secondary_chain(start_pos: vec3f, sun_dir: vec3f, wl_idx: u32,
                         start_optics: ShellOptics, rng: ptr<function, u32>) -> f32 {
    let local_up = normalize(start_pos);
    let surface_radius = EARTH_RADIUS_M;

    let xi_mix = xorshift_f32(rng);
    var dir: vec3f;
    if xi_mix < 0.5 {
        var cos_theta_init: f32;
        if xorshift_f32(rng) < start_optics.rayleigh_fraction {
            cos_theta_init = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_init = sample_henyey_greenstein(xorshift_f32(rng), start_optics.asymmetry);
        }
        let phi_init = 2.0 * PI * xorshift_f32(rng);
        dir = scatter_direction(sun_dir, cos_theta_init, phi_init);
    } else {
        dir = sample_hemisphere(local_up, rng);
    }

    var pos = start_pos;
    var current_dir = dir;
    var weight = start_optics.ssa;
    var total_contribution = kahan_init();

    for (var bounce = 0u; bounce < HYBRID_MAX_BOUNCES; bounce++) {
        let r = length(pos);
        let sidx = shell_index_binary(r);
        if sidx < 0 { break; }

        let us = u32(sidx);
        let sh = read_shell(us);
        let op = read_optics(us, wl_idx);

        if op.extinction < 1e-20 {
            let bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
            if !bnd.found { break; }
            pos = radial_nudge(pos + current_dir * bnd.dist, bnd.is_outward);
            continue;
        }

        let xi = xorshift_f32(rng);
        let free_path = -log(1.0 - xi + 1e-30) / op.extinction;

        let bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
        if !bnd.found { break; }

        if free_path >= bnd.dist {
            let boundary_pos = pos + current_dir * bnd.dist;

            if !bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M {
                let albedo = read_albedo(wl_idx);
                weight *= albedo;
                if weight < 1e-30 { break; }
                let normal = normalize(boundary_pos);
                current_dir = sample_hemisphere(normal, rng);
                pos = radial_nudge(boundary_pos, true);
                continue;
            }

            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        pos = pos + current_dir * free_path;

        let t_sun_sec = shadow_ray_transmittance(pos, sun_dir, wl_idx);
        if t_sun_sec > 1e-30 {
            let cos_angle = dot(sun_dir, -current_dir);
            let phase = mixed_phase(cos_angle, op);
            total_contribution = kahan_add(total_contribution,
                weight * t_sun_sec * phase / (4.0 * PI));
        }

        weight *= op.ssa;

        if weight < 0.01 {
            let xi_rr = xorshift_f32(rng);
            if xi_rr > 0.1 { break; }
            weight /= 0.1;
        }

        var cos_theta_s: f32;
        if xorshift_f32(rng) < op.rayleigh_fraction {
            cos_theta_s = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_s = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        let phi = 2.0 * PI * xorshift_f32(rng);
        current_dir = scatter_direction(current_dir, cos_theta_s, phi);
    }

    return kahan_result(total_contribution);
}

// ============================================================================
// Kernel 3: hybrid_scatter (REPARALLELIZED)
//
// 1 workgroup per wavelength, 256 threads per workgroup.
// Each thread handles one LOS step + secondary chains.
// Reduction: subgroupAdd() + shared memory.
// ============================================================================

var<workgroup> shared_ext_ds: array<f32, 256>;
var<workgroup> shared_sums: array<f32, 8>; // 256/32 = 8 subgroups max

@compute @workgroup_size(256)
fn hybrid_scatter(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(subgroup_invocation_id) simd_lane: u32,
    @builtin(subgroup_id) simd_id: u32,
) {
    let wl_idx = wgid.x;
    let step_idx = lid.x;
    let num_wl = atm_num_wavelengths_val();
    if wl_idx >= num_wl { return; }

    let observer_pos = read_observer();
    let view_dir = read_view_dir();
    let sun_dir = read_sun_dir();
    let secondary_rays_count = read_secondary_rays_val();

    let toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    let surface_radius = EARTH_RADIUS_M;

    // LOS geometry
    let toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    let valid_los = toa_hit.hit && toa_hit.t_far > 0.0;

    var num_steps = 0u;
    var ds = 0.0f;

    if valid_los {
        let los_max = toa_hit.t_far;
        let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
        let hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3 && ground_hit.t_near < los_max;
        var los_end = los_max;
        if hits_ground { los_end = ground_hit.t_near; }
        if los_end > 0.0 {
            num_steps = min(HYBRID_LOS_STEPS, u32(los_end / 500.0) + 20u);
            ds = los_end / f32(num_steps);
        }
    }

    // Phase 1: per-step ext*ds
    var my_ext_ds = 0.0f;
    var my_beta_scat = 0.0f;
    var scatter_pos = vec3f(0.0);
    var my_sidx: i32 = -1;
    var my_op = ShellOptics(0.0, 0.0, 0.0, 0.0);

    if valid_los && step_idx < num_steps {
        let s = (f32(step_idx) + 0.5) * ds;
        scatter_pos = observer_pos + view_dir * s;
        let r = length(scatter_pos);

        if r <= toa_radius && r >= surface_radius {
            my_sidx = shell_index_binary(r);
            if my_sidx >= 0 {
                my_op = read_optics(u32(my_sidx), wl_idx);
                my_ext_ds = my_op.extinction * ds;
                my_beta_scat = my_op.extinction * my_op.ssa;
            }
        }
    }

    shared_ext_ds[step_idx] = my_ext_ds;
    workgroupBarrier();

    // Phase 2: tau_obs
    var tau_obs = 0.0f;
    for (var i = 0u; i < step_idx && i < num_steps; i++) {
        tau_obs += shared_ext_ds[i];
    }
    // exp(-(A+B)) = exp(-A)*exp(-B) avoids f32 precision loss
    let t_obs = exp(-tau_obs) * exp(-my_ext_ds * 0.5);

    // Phase 3: per-step contribution
    var contribution = 0.0f;

    if valid_los && step_idx < num_steps && my_sidx >= 0
        && my_beta_scat > 1e-30 && t_obs > 1e-30
    {
        let seed_lo = read_rng_seed_lo();
        var rng = seed_lo ^ (wl_idx * 2654435761u) ^ (step_idx * 2246822519u);
        xorshift_f32(&rng);

        let t_sun = shadow_ray_transmittance(scatter_pos, sun_dir, wl_idx);
        let cos_theta_1 = dot(sun_dir, -view_dir);
        let phase_1 = mixed_phase(cos_theta_1, my_op);
        let di_single = my_beta_scat * phase_1 / (4.0 * PI) * t_sun * t_obs * ds;
        contribution += di_single;

        if secondary_rays_count > 0u {
            var mc_sum = kahan_init();
            for (var ray = 0u; ray < secondary_rays_count; ray++) {
                mc_sum = kahan_add(mc_sum,
                    trace_secondary_chain(scatter_pos, sun_dir, wl_idx, my_op, &rng));
            }
            let mc_avg = kahan_result(mc_sum) / f32(secondary_rays_count);
            let di_multi = my_beta_scat * t_obs * ds * mc_avg;
            contribution += di_multi;
        }
    }

    // Phase 4: two-level reduction
    let simd_total = subgroupAdd(contribution);

    if simd_lane == 0u {
        shared_sums[simd_id] = simd_total;
    }
    workgroupBarrier();

    if step_idx == 0u {
        var total = 0.0f;
        // Sum up to 8 subgroup partial sums
        for (var i = 0u; i < 8u; i++) {
            total += shared_sums[i];
        }
        output_buf[wl_idx] = total;
    }
}

// ============================================================================
// Kernel 4: garstang_zenith
// ============================================================================

@compute @workgroup_size(256)
fn garstang_zenith(@builtin(global_invocation_id) gid: vec3u) {
    let tid = gid.x;

    let observer_elevation = params_buf[0u];
    let aod_550 = params_buf[1u];
    let uplight_fraction = params_buf[2u];
    let ground_reflectance = params_buf[3u];
    let wavelength_nm = params_buf[4u];
    let altitude_steps = u32(params_buf[5u]);
    let max_altitude = params_buf[6u];
    let num_sources = u32(params_buf[7u]);

    if tid >= num_sources { return; }

    let base = tid * 8u;
    let distance_m = input_buf[base + 0u];
    let source_rad = input_buf[base + 2u];

    if distance_m < 1.0 {
        output_buf[tid] = 0.0;
        return;
    }

    let wl_ratio = 550.0 / wavelength_nm;
    let rayleigh_tau = TAU_RAYLEIGH_550 * wl_ratio * wl_ratio * wl_ratio * wl_ratio;
    let aerosol_tau = aod_550 * pow(wl_ratio, 1.3);

    let effective_up = uplight_fraction + ground_reflectance * 0.5;
    let source_intensity = source_rad * effective_up;

    let dh = max_altitude / f32(altitude_steps);
    var integral = kahan_init();
    let d = distance_m;

    for (var step = 0u; step < altitude_steps; step++) {
        let h = (f32(step) + 0.5) * dh;
        if h < observer_elevation { continue; }

        let r_src_to_scat = sqrt(d * d + h * h);
        let theta_scatter = PI - atan2(d, max(h, 1e-6));

        let n_rayleigh = rayleigh_tau / H_RAYLEIGH * exp(-h / H_RAYLEIGH);
        let n_aerosol = aerosol_tau / H_AEROSOL * exp(-h / H_AEROSOL);
        let sigma_total = n_rayleigh + n_aerosol;

        let cos_scatter = cos(theta_scatter);
        let p_rayleigh = 3.0 / (16.0 * PI) * (1.0 + cos_scatter * cos_scatter);
        let g = 0.7f;
        let g2 = g * g;
        let denom = 1.0 + g2 - 2.0 * g * cos_scatter;
        let p_mie = (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));

        var f_rayleigh = 0.5f;
        if sigma_total > 0.0 { f_rayleigh = n_rayleigh / sigma_total; }
        let p_avg = f_rayleigh * p_rayleigh + (1.0 - f_rayleigh) * p_mie;

        let path_len = r_src_to_scat;
        var tau_slant = 0.0f;
        if h > 1.0 {
            let n0_r = rayleigh_tau / H_RAYLEIGH;
            tau_slant += n0_r * path_len * H_RAYLEIGH / h * (1.0 - exp(-h / H_RAYLEIGH));
            let n0_a = aerosol_tau / H_AEROSOL;
            tau_slant += n0_a * path_len * H_AEROSOL / h * (1.0 - exp(-h / H_AEROSOL));
        } else {
            tau_slant = (rayleigh_tau / H_RAYLEIGH + aerosol_tau / H_AEROSOL) * path_len;
        }

        let tau_vert = rayleigh_tau * (exp(-observer_elevation / H_RAYLEIGH) - exp(-h / H_RAYLEIGH))
                     + aerosol_tau * (exp(-observer_elevation / H_AEROSOL) - exp(-h / H_AEROSOL));

        let extinction = exp(-tau_slant - tau_vert);
        let r2 = r_src_to_scat * r_src_to_scat;

        let di = source_intensity / (4.0 * PI * r2) * sigma_total * p_avg * extinction * dh;
        integral = kahan_add(integral, di);
    }

    output_buf[tid] = kahan_result(integral);
}
