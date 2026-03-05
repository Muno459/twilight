// Twilight MCRT - WGSL compute shaders for WebGPU
//
// Four compute kernels for GPU-accelerated twilight radiative transfer:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_scatter            - LOS + secondary MC chains
//   4. garstang_zenith           - Light pollution skyglow
//
// All four kernels are in one module with separate @compute entry points.
// Buffer layout matches crates/twilight-gpu/src/buffers.rs exactly.
// All physics ported from twilight-core (f64) to f32 GPU precision.
// Uses xorshift32 RNG (WGSL has no native u64).

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

const RUSSIAN_ROULETTE_WEIGHT: f32 = 0.01;
const RUSSIAN_ROULETTE_SURVIVE: f32 = 0.1;

// Atmosphere buffer offsets (must match buffers.rs atm_offsets)
const ATM_NUM_SHELLS: u32 = 2u;
const ATM_NUM_WAVELENGTHS: u32 = 3u;
const ATM_SHELLS_START: u32 = 4u;
const ATM_SHELL_STRIDE: u32 = 4u;
const ATM_OPTICS_START: u32 = 260u;   // 4 + 4*64
const ATM_OPTICS_STRIDE: u32 = 4u;
const ATM_ALBEDO_START: u32 = 16708u; // 16644 + 64

// Garstang constants
const H_RAYLEIGH: f32 = 8500.0;
const H_AEROSOL: f32 = 1500.0;
const TAU_RAYLEIGH_550: f32 = 0.0962;

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

fn read_secondary_rays() -> u32 {
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
    let b = 2.0 * dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - 4.0 * a * c;

    if disc < 0.0 {
        return RaySphereHit(0.0, 0.0, false);
    }

    let sqrt_disc = sqrt(disc);
    let inv_2a = 0.5 / a;
    let t_near = (-b - sqrt_disc) * inv_2a;
    let t_far = (-b + sqrt_disc) * inv_2a;
    return RaySphereHit(t_near, t_far, true);
}

// ============================================================================
// Shell index lookup
// ============================================================================

fn shell_index(r: f32) -> i32 {
    let ns = atm_num_shells_val();
    for (var s = 0u; s < ns; s++) {
        let sh = read_shell(s);
        if r >= sh.r_inner && r < sh.r_outer {
            return i32(s);
        }
    }
    if ns > 0u {
        let last = read_shell(ns - 1u);
        if r >= last.r_inner && r <= last.r_outer + 1.0 {
            return i32(ns - 1u);
        }
    }
    return -1;
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
// Ray path through shell (analytical)
// ============================================================================

fn ray_path_through_shell(origin: vec3f, dir: vec3f,
                          r_inner: f32, r_outer: f32, t_max: f32) -> f32 {
    let outer = ray_sphere_intersect(origin, dir, r_outer);
    if !outer.hit { return 0.0; }

    let t0_outer = max(outer.t_near, 0.0);
    let t1_outer = min(outer.t_far, t_max);
    if t1_outer <= t0_outer + 1e-6 { return 0.0; }

    let inner = ray_sphere_intersect(origin, dir, r_inner);
    if !inner.hit {
        return t1_outer - t0_outer;
    }

    let t0_inner = max(inner.t_near, 0.0);
    let t1_inner = min(inner.t_far, t_max);
    if t1_inner <= t0_inner + 1e-6 {
        return t1_outer - t0_outer;
    }

    var total = 0.0f;
    let seg1_end = min(t1_outer, t0_inner);
    if seg1_end > t0_outer { total += seg1_end - t0_outer; }
    let seg2_start = max(t0_outer, t1_inner);
    if t1_outer > seg2_start { total += t1_outer - seg2_start; }

    return total;
}

// ============================================================================
// Shadow ray transmittance
// ============================================================================

fn shadow_ray_transmittance(start_pos: vec3f, sun_dir: vec3f, wl_idx: u32) -> f32 {
    let ns = atm_num_shells_val();
    let toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    let surface_radius = EARTH_RADIUS_M;

    let toa_hit = ray_sphere_intersect(start_pos, sun_dir, toa_radius);
    if !toa_hit.hit || toa_hit.t_far <= 0.0 { return 0.0; }
    let ray_max = toa_hit.t_far;

    let ground_hit = ray_sphere_intersect(start_pos, sun_dir, surface_radius);
    if ground_hit.hit && ground_hit.t_near > 1e-3 && ground_hit.t_near < ray_max {
        return 0.0;
    }

    var tau = 0.0f;
    for (var s = 0u; s < ns; s++) {
        let sh = read_shell(s);
        let path = ray_path_through_shell(start_pos, sun_dir, sh.r_inner, sh.r_outer, ray_max);
        if path > 0.0 {
            let op = read_optics(s, wl_idx);
            tau += op.extinction * path;
            if tau > 50.0 { return 0.0; }
        }
    }
    return exp(-tau);
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
// Next shell boundary
// ============================================================================

fn next_shell_boundary(pos: vec3f, dir: vec3f, r_inner: f32, r_outer: f32) -> ShellBoundary {
    var result = ShellBoundary(1e30, true, false);

    let outer = ray_sphere_intersect(pos, dir, r_outer);
    if outer.hit {
        if outer.t_near > 1e-10 {
            let inner = ray_sphere_intersect(pos, dir, r_inner);
            if inner.hit && inner.t_near > 1e-10 && inner.t_near < outer.t_near {
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
        if outer.t_far > 1e-10 {
            let inner = ray_sphere_intersect(pos, dir, r_inner);
            if inner.hit && inner.t_near > 1e-10 && inner.t_near < outer.t_far {
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
    if inner.hit && inner.t_near > 1e-10 {
        result.dist = inner.t_near;
        result.is_outward = false;
        result.found = true;
    }
    return result;
}

// ============================================================================
// Kernel 1: single_scatter_spectrum
//
// One thread per wavelength. Full LOS integration with analytical shadow rays.
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

    var radiance = 0.0f;
    var tau_obs = 0.0f;
    let cos_theta = dot(sun_dir, -view_dir);

    for (var step = 0u; step < num_steps; step++) {
        let s = (f32(step) + 0.5) * ds;
        let scatter_pos = observer_pos + view_dir * s;
        let r = length(scatter_pos);

        if r > toa_radius || r < surface_radius { continue; }

        let sidx = shell_index(r);
        if sidx < 0 { continue; }

        let op = read_optics(u32(sidx), wl_idx);
        let beta_scat = op.extinction * op.ssa;

        if beta_scat < 1e-30 {
            tau_obs += op.extinction * ds;
            continue;
        }

        let tau_obs_mid = tau_obs + op.extinction * ds * 0.5;
        let t_obs = exp(-tau_obs_mid);
        if t_obs < 1e-30 { break; }

        let t_sun = shadow_ray_transmittance(scatter_pos, sun_dir, wl_idx);
        if t_sun < 1e-30 {
            tau_obs += op.extinction * ds;
            continue;
        }

        let phase = mixed_phase(cos_theta, op);
        let di = beta_scat * phase / (4.0 * PI) * t_sun * t_obs * ds;
        radiance += di;
        tau_obs += op.extinction * ds;
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
                let t_obs_ground = exp(-tau_obs);
                radiance += albedo / PI * cos_sun_incidence * t_sun_ground * t_obs_ground;
            }
        }
    }

    output_buf[tid] = radiance;
}

// ============================================================================
// Kernel 2: mcrt_trace_photon
//
// One thread per (wavelength, photon) pair.
// tid = wl_idx * photons_per_wl + photon_idx
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

    // Unique seed per (wavelength, photon) pair
    let base_seed = read_rng_seed_lo();
    var rng = base_seed + wl_idx;
    rng = rng * 1664525u + photon_idx;
    rng = rng * 1013904223u + 1u;

    let surface_radius = EARTH_RADIUS_M;

    var pos = observer_pos;
    var dir = view_dir;
    var weight = 1.0f;
    var result_weight = 0.0f;

    for (var bounce = 0u; bounce < MAX_SCATTERS; bounce++) {
        let r = length(pos);
        let sidx = shell_index(r);
        if sidx < 0 { break; }

        let sh = read_shell(u32(sidx));
        let op = read_optics(u32(sidx), wl_idx);

        if op.extinction < 1e-20 {
            let bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
            if !bnd.found { break; }
            pos = pos + dir * (bnd.dist + 1e-3);
            continue;
        }

        let xi = xorshift_f32(&rng);
        let free_path = -log(1.0 - xi + 1e-30) / op.extinction;

        let bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
        if !bnd.found { break; }

        if free_path >= bnd.dist {
            pos = pos + dir * (bnd.dist + 1e-3);
            if !bnd.is_outward && length(pos) <= surface_radius + 1.0 {
                let albedo = read_albedo(wl_idx);
                weight *= albedo;
                let normal = normalize(pos);
                dir = sample_hemisphere(normal, &rng);
            }
            continue;
        }

        pos = pos + dir * free_path;

        // NEE
        let t_sun = shadow_ray_transmittance(pos, sun_dir, wl_idx);
        if t_sun > 1e-30 {
            let cos_angle = dot(sun_dir, -dir);
            let phase = mixed_phase(cos_angle, op);
            result_weight += weight * t_sun * phase / (4.0 * PI);
        }

        weight *= op.ssa;

        if weight < RUSSIAN_ROULETTE_WEIGHT {
            let xi_rr = xorshift_f32(&rng);
            if xi_rr > RUSSIAN_ROULETTE_SURVIVE { break; }
            weight /= RUSSIAN_ROULETTE_SURVIVE;
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

    output_buf[tid] = result_weight;
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
    var total_contribution = 0.0f;

    for (var bounce = 0u; bounce < HYBRID_MAX_BOUNCES; bounce++) {
        let r = length(pos);
        let sidx = shell_index(r);
        if sidx < 0 { break; }

        let sh = read_shell(u32(sidx));
        let op = read_optics(u32(sidx), wl_idx);

        if op.extinction < 1e-20 {
            let bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
            if !bnd.found { break; }
            pos = pos + current_dir * (bnd.dist + 1e-3);
            continue;
        }

        let xi = xorshift_f32(rng);
        let free_path = -log(1.0 - xi + 1e-30) / op.extinction;

        let bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
        if !bnd.found { break; }

        if free_path >= bnd.dist {
            pos = pos + current_dir * (bnd.dist + 1e-3);
            if !bnd.is_outward && length(pos) <= surface_radius + 1.0 {
                let albedo = read_albedo(wl_idx);
                weight *= albedo;
                if weight < 1e-30 { break; }
                let normal = normalize(pos);
                current_dir = sample_hemisphere(normal, rng);
            }
            continue;
        }

        pos = pos + current_dir * free_path;

        let t_sun_sec = shadow_ray_transmittance(pos, sun_dir, wl_idx);
        if t_sun_sec > 1e-30 {
            let cos_angle = dot(sun_dir, -current_dir);
            let phase = mixed_phase(cos_angle, op);
            total_contribution += weight * t_sun_sec * phase / (4.0 * PI);
        }

        weight *= op.ssa;

        if weight < RUSSIAN_ROULETTE_WEIGHT {
            let xi_rr = xorshift_f32(rng);
            if xi_rr > RUSSIAN_ROULETTE_SURVIVE { break; }
            weight /= RUSSIAN_ROULETTE_SURVIVE;
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

    return total_contribution;
}

// ============================================================================
// Kernel 3: hybrid_scatter
//
// One thread per wavelength. LOS integration + secondary MC chains.
// ============================================================================

@compute @workgroup_size(256)
fn hybrid_scatter(@builtin(global_invocation_id) gid: vec3u) {
    let tid = gid.x;
    let num_wl = atm_num_wavelengths_val();
    if tid >= num_wl { return; }

    let wl_idx = tid;
    let observer_pos = read_observer();
    let view_dir = read_view_dir();
    let sun_dir = read_sun_dir();
    let secondary_rays_count = read_secondary_rays();

    var rng = read_rng_seed_lo() + wl_idx;
    rng = rng * 1664525u + 1u;

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

    let num_steps = min(HYBRID_LOS_STEPS, u32(los_end / 500.0) + 20u);
    let ds = los_end / f32(num_steps);

    var radiance = 0.0f;
    var tau_obs = 0.0f;

    for (var step = 0u; step < num_steps; step++) {
        let s = (f32(step) + 0.5) * ds;
        let scatter_pos = observer_pos + view_dir * s;
        let r = length(scatter_pos);

        if r > toa_radius || r < surface_radius { continue; }

        let sidx = shell_index(r);
        if sidx < 0 { continue; }

        let op = read_optics(u32(sidx), wl_idx);
        let beta_scat = op.extinction * op.ssa;

        if beta_scat < 1e-30 {
            tau_obs += op.extinction * ds;
            continue;
        }

        let tau_obs_mid = tau_obs + op.extinction * ds * 0.5;
        let t_obs = exp(-tau_obs_mid);
        if t_obs < 1e-30 { break; }

        let t_sun = shadow_ray_transmittance(scatter_pos, sun_dir, wl_idx);
        let cos_theta_1 = dot(sun_dir, -view_dir);
        let phase_1 = mixed_phase(cos_theta_1, op);
        let di_single = beta_scat * phase_1 / (4.0 * PI) * t_sun * t_obs * ds;
        radiance += di_single;

        if secondary_rays_count > 0u {
            var mc_sum = 0.0f;
            for (var ray = 0u; ray < secondary_rays_count; ray++) {
                mc_sum += trace_secondary_chain(scatter_pos, sun_dir, wl_idx, op, &rng);
            }
            let mc_avg = mc_sum / f32(secondary_rays_count);
            let di_multi = beta_scat * t_obs * ds * mc_avg;
            radiance += di_multi;
        }

        tau_obs += op.extinction * ds;
    }

    output_buf[tid] = radiance;
}

// ============================================================================
// Kernel 4: garstang_zenith
//
// One thread per light source. Uses separate buffer layout.
// Input buffer: sources (8 floats per source)
// Params buffer: config (8 floats)
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
    var integral = 0.0f;
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
        integral += di;
    }

    output_buf[tid] = integral;
}
