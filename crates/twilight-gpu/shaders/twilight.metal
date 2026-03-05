// Twilight MCRT - Metal Shading Language compute kernels
//
// Four compute kernels for GPU-accelerated twilight radiative transfer:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_scatter            - LOS + secondary MC chains
//   4. garstang_zenith           - Light pollution skyglow
//
// Buffer layout matches crates/twilight-gpu/src/buffers.rs exactly.
// All physics ported from twilight-core (f64) to f32 GPU precision.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant float PI = 3.14159265358979323846f;
constant float EARTH_RADIUS_M = 6371000.0f;
constant float TOA_ALTITUDE_M = 100000.0f;

constant uint MAX_WAVELENGTHS = 64;
constant uint MAX_LOS_STEPS = 200;
constant uint MAX_SCATTERS = 100;
constant uint HYBRID_LOS_STEPS = 200;
constant uint HYBRID_MAX_BOUNCES = 50;

constant float RUSSIAN_ROULETTE_WEIGHT = 0.01f;
constant float RUSSIAN_ROULETTE_SURVIVE = 0.1f;

// Buffer header magic
// Atmosphere buffer offsets (must match buffers.rs atm_offsets)
constant uint ATM_NUM_SHELLS      = 2;
constant uint ATM_NUM_WAVELENGTHS = 3;
constant uint ATM_SHELLS_START    = 4;
constant uint ATM_SHELL_STRIDE    = 4;
constant uint ATM_OPTICS_START    = 260;  // 4 + 4*64
constant uint ATM_OPTICS_STRIDE   = 4;
constant uint ATM_ALBEDO_START    = 16708;   // 16644 + 64

// Garstang constants
constant float H_RAYLEIGH = 8500.0f;
constant float H_AEROSOL  = 1500.0f;
constant float TAU_RAYLEIGH_550 = 0.0962f;

// ============================================================================
// Buffer accessor helpers
// ============================================================================

struct ShellGeom {
    float r_inner;
    float r_outer;
    float altitude_mid;
    float thickness;
};

struct ShellOptics {
    float extinction;
    float ssa;
    float asymmetry;
    float rayleigh_fraction;
};

inline uint read_num_shells(device const float* atm) {
    return as_type<uint>(atm[ATM_NUM_SHELLS]) > 0
        ? uint(atm[ATM_NUM_SHELLS])
        : as_type<uint>(atm[ATM_NUM_SHELLS]);
}

inline uint atm_num_shells(device const float* atm) {
    return uint(atm[ATM_NUM_SHELLS]);
}

inline uint atm_num_wavelengths(device const float* atm) {
    return uint(atm[ATM_NUM_WAVELENGTHS]);
}

inline ShellGeom read_shell(device const float* atm, uint shell_idx) {
    uint base = ATM_SHELLS_START + shell_idx * ATM_SHELL_STRIDE;
    return ShellGeom{
        atm[base + 0],
        atm[base + 1],
        atm[base + 2],
        atm[base + 3]
    };
}

inline ShellOptics read_optics(device const float* atm, uint shell_idx, uint wl_idx) {
    uint idx = shell_idx * MAX_WAVELENGTHS + wl_idx;
    uint base = ATM_OPTICS_START + idx * ATM_OPTICS_STRIDE;
    return ShellOptics{
        atm[base + 0],
        atm[base + 1],
        atm[base + 2],
        atm[base + 3]
    };
}

inline float read_albedo(device const float* atm, uint wl_idx) {
    return atm[ATM_ALBEDO_START + wl_idx];
}

// Dispatch params: 4 x vec4
// vec4(obs_x, obs_y, obs_z, pad)
// vec4(view_x, view_y, view_z, pad)
// vec4(sun_x, sun_y, sun_z, pad)
// vec4(photons_bits, secondary_bits, seed_lo_bits, seed_hi_bits)
inline float3 read_observer(device const float* params) {
    return float3(params[0], params[1], params[2]);
}
inline float3 read_view_dir(device const float* params) {
    return float3(params[4], params[5], params[6]);
}
inline float3 read_sun_dir(device const float* params) {
    return float3(params[8], params[9], params[10]);
}
inline uint read_photons_per_wl(device const float* params) {
    return as_type<uint>(params[12]);
}
inline uint read_secondary_rays(device const float* params) {
    return as_type<uint>(params[13]);
}
inline ulong read_rng_seed(device const float* params) {
    uint lo = as_type<uint>(params[14]);
    uint hi = as_type<uint>(params[15]);
    return ulong(lo) | (ulong(hi) << 32);
}

// ============================================================================
// Math utilities
// ============================================================================

inline float len3(float3 v) { return length(v); }
inline float dot3(float3 a, float3 b) { return dot(a, b); }

// ============================================================================
// xorshift64 RNG (Metal supports ulong natively on Apple Silicon)
// ============================================================================

inline float xorshift_f32(thread ulong &state) {
    ulong x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    // Convert to f32 in [0, 1): use top 24 bits for full mantissa precision
    return float(x >> 40) * (1.0f / float(1ul << 24));
}

// ============================================================================
// Ray-sphere intersection
// ============================================================================

struct RaySphereHit {
    float t_near;
    float t_far;
    bool hit;
};

inline RaySphereHit ray_sphere_intersect(float3 origin, float3 dir, float radius) {
    float a = dot3(dir, dir);
    float b = 2.0f * dot3(origin, dir);
    float c = dot3(origin, origin) - radius * radius;
    float disc = b * b - 4.0f * a * c;

    RaySphereHit result;
    if (disc < 0.0f) {
        result.hit = false;
        result.t_near = 0.0f;
        result.t_far = 0.0f;
        return result;
    }

    float sqrt_disc = sqrt(disc);
    float inv_2a = 0.5f / a;
    result.t_near = (-b - sqrt_disc) * inv_2a;
    result.t_far  = (-b + sqrt_disc) * inv_2a;
    result.hit = true;
    return result;
}

// ============================================================================
// Shell index lookup
// ============================================================================

inline int shell_index(device const float* atm, float r) {
    uint ns = atm_num_shells(atm);
    for (uint s = 0; s < ns; s++) {
        ShellGeom sh = read_shell(atm, s);
        if (r >= sh.r_inner && r < sh.r_outer) {
            return int(s);
        }
    }
    // Check if r is exactly at the outer boundary of the last shell
    if (ns > 0) {
        ShellGeom last = read_shell(atm, ns - 1);
        if (r >= last.r_inner && r <= last.r_outer + 1.0f) {
            return int(ns - 1);
        }
    }
    return -1;
}

// ============================================================================
// Phase functions
// ============================================================================

inline float rayleigh_phase(float cos_theta) {
    return 0.75f * (1.0f + cos_theta * cos_theta);
}

inline float henyey_greenstein_phase(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    return (1.0f - g2) / (denom * sqrt(denom));
}

inline float mixed_phase(float cos_theta, ShellOptics op) {
    if (op.rayleigh_fraction > 0.99f) {
        return rayleigh_phase(cos_theta);
    }
    return op.rayleigh_fraction * rayleigh_phase(cos_theta)
         + (1.0f - op.rayleigh_fraction) * henyey_greenstein_phase(cos_theta, op.asymmetry);
}

// ============================================================================
// Ray path through shell (analytical)
// ============================================================================

float ray_path_through_shell(float3 origin, float3 dir,
                             float r_inner, float r_outer, float t_max) {
    // Outer sphere intersection
    RaySphereHit outer = ray_sphere_intersect(origin, dir, r_outer);
    if (!outer.hit) return 0.0f;

    float t0_outer = max(outer.t_near, 0.0f);
    float t1_outer = min(outer.t_far, t_max);
    if (t1_outer <= t0_outer + 1e-6f) return 0.0f;

    // Inner sphere intersection (to subtract)
    RaySphereHit inner = ray_sphere_intersect(origin, dir, r_inner);
    if (!inner.hit) {
        return t1_outer - t0_outer;
    }

    float t0_inner = max(inner.t_near, 0.0f);
    float t1_inner = min(inner.t_far, t_max);
    if (t1_inner <= t0_inner + 1e-6f) {
        return t1_outer - t0_outer;
    }

    // Shell = outer - inner
    float total = 0.0f;
    // Segment before inner
    float seg1_end = min(t1_outer, t0_inner);
    if (seg1_end > t0_outer) total += seg1_end - t0_outer;
    // Segment after inner
    float seg2_start = max(t0_outer, t1_inner);
    if (t1_outer > seg2_start) total += t1_outer - seg2_start;

    return total;
}

// ============================================================================
// Shadow ray transmittance (analytical, shell-by-shell)
// ============================================================================

float shadow_ray_transmittance(device const float* atm, float3 start_pos,
                                float3 sun_dir, uint wl_idx) {
    uint ns = atm_num_shells(atm);
    float toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    float surface_radius = EARTH_RADIUS_M;

    // Find where shadow ray exits atmosphere
    RaySphereHit toa_hit = ray_sphere_intersect(start_pos, sun_dir, toa_radius);
    if (!toa_hit.hit || toa_hit.t_far <= 0.0f) return 0.0f;
    float ray_max = toa_hit.t_far;

    // Check if shadow ray hits the ground
    RaySphereHit ground_hit = ray_sphere_intersect(start_pos, sun_dir, surface_radius);
    if (ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < ray_max) {
        return 0.0f;
    }

    // Analytical integration: exact path length through each shell
    float tau = 0.0f;
    for (uint s = 0; s < ns; s++) {
        ShellGeom sh = read_shell(atm, s);
        float path = ray_path_through_shell(start_pos, sun_dir, sh.r_inner, sh.r_outer, ray_max);
        if (path > 0.0f) {
            ShellOptics op = read_optics(atm, s, wl_idx);
            tau += op.extinction * path;
            if (tau > 50.0f) return 0.0f;
        }
    }
    return exp(-tau);
}

// ============================================================================
// Sampling functions
// ============================================================================

inline float sample_rayleigh_analytic(float xi) {
    // Cardano's formula for depressed cubic: mu^3 + 3*mu - q = 0
    float q = 8.0f * xi - 4.0f;
    float disc = q * q * 0.25f + 1.0f;
    float sqrt_disc = sqrt(disc);
    // cbrt is available in MSL via precise::cbrt or just pow with sign handling
    float a_val = -q * 0.5f + sqrt_disc;
    float b_val = -q * 0.5f - sqrt_disc;
    float u = (a_val >= 0.0f) ? pow(a_val, 1.0f/3.0f) : -pow(-a_val, 1.0f/3.0f);
    float v = (b_val >= 0.0f) ? pow(b_val, 1.0f/3.0f) : -pow(-b_val, 1.0f/3.0f);
    float mu = u + v;
    return clamp(mu, -1.0f, 1.0f);
}

inline float sample_henyey_greenstein(float xi, float g) {
    if (abs(g) < 1e-6f) {
        return 2.0f * xi - 1.0f;
    }
    float g2 = g * g;
    float s = (1.0f - g2) / (1.0f - g + 2.0f * g * xi);
    float mu = (1.0f + g2 - s * s) / (2.0f * g);
    return clamp(mu, -1.0f, 1.0f);
}

float3 scatter_direction(float3 dir, float cos_theta, float phi) {
    float sin_theta = sqrt(max(1.0f - cos_theta * cos_theta, 0.0f));
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    float3 w = dir;
    float3 up = (abs(w.z) < 0.9f) ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 u_vec = normalize(cross(w, up));
    float3 v_vec = cross(w, u_vec);

    float3 new_dir = sin_theta * cos_phi * u_vec
                   + sin_theta * sin_phi * v_vec
                   + cos_theta * w;
    return normalize(new_dir);
}

float3 sample_hemisphere(float3 normal, thread ulong &rng) {
    float xi1 = xorshift_f32(rng);
    float xi2 = xorshift_f32(rng);
    float cos_theta = sqrt(xi1);
    float phi = 2.0f * PI * xi2;
    return scatter_direction(normal, cos_theta, phi);
}

// ============================================================================
// Next shell boundary
// ============================================================================

struct ShellBoundary {
    float dist;
    bool is_outward;
    bool found;
};

ShellBoundary next_shell_boundary(float3 pos, float3 dir, float r_inner, float r_outer) {
    ShellBoundary result;
    result.found = false;
    result.dist = 1e30f;
    result.is_outward = true;

    RaySphereHit outer = ray_sphere_intersect(pos, dir, r_outer);
    if (outer.hit) {
        if (outer.t_near > 1e-10f) {
            // Check inner first
            RaySphereHit inner = ray_sphere_intersect(pos, dir, r_inner);
            if (inner.hit && inner.t_near > 1e-10f && inner.t_near < outer.t_near) {
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
        if (outer.t_far > 1e-10f) {
            RaySphereHit inner = ray_sphere_intersect(pos, dir, r_inner);
            if (inner.hit && inner.t_near > 1e-10f && inner.t_near < outer.t_far) {
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

    // Fallback: inner only
    RaySphereHit inner = ray_sphere_intersect(pos, dir, r_inner);
    if (inner.hit && inner.t_near > 1e-10f) {
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
// Output: radiance[wl_idx] (f32)
// ============================================================================

kernel void single_scatter_spectrum(
    device const float* atm       [[buffer(0)]],
    device const float* params    [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    uint                tid       [[thread_position_in_grid]]
) {
    uint num_wl = atm_num_wavelengths(atm);
    if (tid >= num_wl) return;

    uint wl_idx = tid;
    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);

    float toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    float surface_radius = EARTH_RADIUS_M;

    // Find LOS extent
    RaySphereHit toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    if (!toa_hit.hit || toa_hit.t_far <= 0.0f) {
        output[tid] = 0.0f;
        return;
    }
    float los_max = toa_hit.t_far;

    RaySphereHit ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    bool hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < los_max;
    float los_end = hits_ground ? ground_hit.t_near : los_max;

    if (los_end <= 0.0f) {
        output[tid] = 0.0f;
        return;
    }

    uint num_steps = min(MAX_LOS_STEPS, uint(los_end / 500.0f) + 20u);
    float ds = los_end / float(num_steps);

    float radiance = 0.0f;
    float tau_obs = 0.0f;

    float cos_theta = dot3(sun_dir, -view_dir);

    for (uint step = 0; step < num_steps; step++) {
        float s = (float(step) + 0.5f) * ds;
        float3 scatter_pos = observer_pos + view_dir * s;
        float r = len3(scatter_pos);

        if (r > toa_radius || r < surface_radius) continue;

        int sidx = shell_index(atm, r);
        if (sidx < 0) continue;

        ShellOptics op = read_optics(atm, uint(sidx), wl_idx);
        float beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30f) {
            tau_obs += op.extinction * ds;
            continue;
        }

        float tau_obs_mid = tau_obs + op.extinction * ds * 0.5f;
        float t_obs = exp(-tau_obs_mid);

        if (t_obs < 1e-30f) break;

        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);

        if (t_sun < 1e-30f) {
            tau_obs += op.extinction * ds;
            continue;
        }

        float phase = mixed_phase(cos_theta, op);
        float di = beta_scat * phase / (4.0f * PI) * t_sun * t_obs * ds;
        radiance += di;

        tau_obs += op.extinction * ds;
    }

    // Ground reflection (Lambertian BRDF = albedo / pi)
    if (hits_ground) {
        float albedo = read_albedo(atm, wl_idx);
        if (albedo > 1e-10f) {
            float3 ground_pos = observer_pos + view_dir * los_end;
            float3 ground_normal = normalize(ground_pos);
            float cos_sun_incidence = dot3(sun_dir, ground_normal);

            if (cos_sun_incidence > 0.0f) {
                float t_sun_ground = shadow_ray_transmittance(atm, ground_pos, sun_dir, wl_idx);
                float t_obs_ground = exp(-tau_obs);
                radiance += albedo / PI * cos_sun_incidence * t_sun_ground * t_obs_ground;
            }
        }
    }

    output[tid] = radiance;
}

// ============================================================================
// Kernel 2: mcrt_trace_photon
//
// One thread per (wavelength, photon) pair.
// Thread index: tid = wl_idx * photons_per_wl + photon_idx
// Output: per-thread weight (CPU reduces)
// ============================================================================

kernel void mcrt_trace_photon(
    device const float* atm       [[buffer(0)]],
    device const float* params    [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    uint                tid       [[thread_position_in_grid]]
) {
    uint num_wl = atm_num_wavelengths(atm);
    uint photons_per_wl = read_photons_per_wl(params);
    uint total_threads = num_wl * photons_per_wl;
    if (tid >= total_threads) return;

    uint wl_idx = tid / photons_per_wl;
    uint photon_idx = tid % photons_per_wl;

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);

    // Unique seed per (wavelength, photon) pair
    ulong base_seed = read_rng_seed(params);
    ulong rng = base_seed + ulong(wl_idx);
    rng *= 6364136223846793005ul;
    rng += ulong(photon_idx);
    rng *= 2862933555777941757ul;
    rng += 1ul;

    float surface_radius = EARTH_RADIUS_M;

    float3 pos = observer_pos;
    float3 dir = view_dir;
    float weight = 1.0f;
    float result_weight = 0.0f;

    for (uint bounce = 0; bounce < MAX_SCATTERS; bounce++) {
        float r = len3(pos);
        int sidx = shell_index(atm, r);
        if (sidx < 0) break;

        ShellGeom sh = read_shell(atm, uint(sidx));
        ShellOptics op = read_optics(atm, uint(sidx), wl_idx);

        if (op.extinction < 1e-20f) {
            ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) break;
            pos = pos + dir * (bnd.dist + 1e-3f);
            continue;
        }

        // Sample free path
        float xi = xorshift_f32(rng);
        float free_path = -log(1.0f - xi + 1e-30f) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            // Exit shell without scattering
            pos = pos + dir * (bnd.dist + 1e-3f);

            // Ground reflection
            if (!bnd.is_outward && len3(pos) <= surface_radius + 1.0f) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                float3 normal = normalize(pos);
                dir = sample_hemisphere(normal, rng);
            }
            continue;
        }

        // Scattering event
        pos = pos + dir * free_path;

        // NEE: direct solar contribution
        float t_sun = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun > 1e-30f) {
            float cos_angle = dot3(sun_dir, -dir);
            float phase = mixed_phase(cos_angle, op);
            result_weight += weight * t_sun * phase / (4.0f * PI);
        }

        // Apply SSA
        weight *= op.ssa;

        // Russian roulette
        if (weight < RUSSIAN_ROULETTE_WEIGHT) {
            float xi_rr = xorshift_f32(rng);
            if (xi_rr > RUSSIAN_ROULETTE_SURVIVE) break;
            weight /= RUSSIAN_ROULETTE_SURVIVE;
        }

        // Sample new direction
        float cos_theta;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        dir = scatter_direction(dir, cos_theta, phi);
    }

    output[tid] = result_weight;
}

// ============================================================================
// Secondary chain tracer (used by hybrid_scatter kernel)
// ============================================================================

float trace_secondary_chain(device const float* atm, float3 start_pos,
                            float3 sun_dir, uint wl_idx,
                            ShellOptics start_optics, thread ulong &rng) {
    float3 local_up = normalize(start_pos);
    float surface_radius = EARTH_RADIUS_M;

    // Importance sampling: 50/50 phase-function vs upward-biased
    float xi_mix = xorshift_f32(rng);
    float3 dir;
    if (xi_mix < 0.5f) {
        float cos_theta_init;
        if (xorshift_f32(rng) < start_optics.rayleigh_fraction) {
            cos_theta_init = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_init = sample_henyey_greenstein(xorshift_f32(rng), start_optics.asymmetry);
        }
        float phi_init = 2.0f * PI * xorshift_f32(rng);
        dir = scatter_direction(sun_dir, cos_theta_init, phi_init);
    } else {
        dir = sample_hemisphere(local_up, rng);
    }

    float3 pos = start_pos;
    float3 current_dir = dir;
    float weight = start_optics.ssa;
    float total_contribution = 0.0f;

    for (uint bounce = 0; bounce < HYBRID_MAX_BOUNCES; bounce++) {
        float r = len3(pos);
        int sidx = shell_index(atm, r);
        if (sidx < 0) break;

        ShellGeom sh = read_shell(atm, uint(sidx));
        ShellOptics op = read_optics(atm, uint(sidx), wl_idx);

        if (op.extinction < 1e-20f) {
            ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) break;
            pos = pos + current_dir * (bnd.dist + 1e-3f);
            continue;
        }

        float xi = xorshift_f32(rng);
        float free_path = -log(1.0f - xi + 1e-30f) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            pos = pos + current_dir * (bnd.dist + 1e-3f);
            if (!bnd.is_outward && len3(pos) <= surface_radius + 1.0f) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                if (weight < 1e-30f) break;
                float3 normal = normalize(pos);
                current_dir = sample_hemisphere(normal, rng);
            }
            continue;
        }

        // Scatter event
        pos = pos + current_dir * free_path;

        // NEE
        float t_sun_sec = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun_sec > 1e-30f) {
            float cos_angle = dot3(sun_dir, -current_dir);
            float phase = mixed_phase(cos_angle, op);
            total_contribution += weight * t_sun_sec * phase / (4.0f * PI);
        }

        weight *= op.ssa;

        // Russian roulette
        if (weight < RUSSIAN_ROULETTE_WEIGHT) {
            float xi_rr = xorshift_f32(rng);
            if (xi_rr > RUSSIAN_ROULETTE_SURVIVE) break;
            weight /= RUSSIAN_ROULETTE_SURVIVE;
        }

        // Sample new direction
        float cos_theta;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        current_dir = scatter_direction(current_dir, cos_theta, phi);
    }

    return total_contribution;
}

// ============================================================================
// Kernel 3: hybrid_scatter
//
// One thread per wavelength. LOS integration + secondary MC chains.
// Output: radiance[wl_idx]
// ============================================================================

kernel void hybrid_scatter(
    device const float* atm       [[buffer(0)]],
    device const float* params    [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    uint                tid       [[thread_position_in_grid]]
) {
    uint num_wl = atm_num_wavelengths(atm);
    if (tid >= num_wl) return;

    uint wl_idx = tid;
    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);
    uint secondary_rays = read_secondary_rays(params);

    // Per-wavelength RNG
    ulong rng = read_rng_seed(params) + ulong(wl_idx);
    rng *= 6364136223846793005ul;
    rng += 1ul;

    float toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    float surface_radius = EARTH_RADIUS_M;

    RaySphereHit toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    if (!toa_hit.hit || toa_hit.t_far <= 0.0f) {
        output[tid] = 0.0f;
        return;
    }
    float los_max = toa_hit.t_far;

    RaySphereHit ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    bool hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < los_max;
    float los_end = hits_ground ? ground_hit.t_near : los_max;

    if (los_end <= 0.0f) {
        output[tid] = 0.0f;
        return;
    }

    uint num_steps = min(HYBRID_LOS_STEPS, uint(los_end / 500.0f) + 20u);
    float ds = los_end / float(num_steps);

    float radiance = 0.0f;
    float tau_obs = 0.0f;

    for (uint step = 0; step < num_steps; step++) {
        float s = (float(step) + 0.5f) * ds;
        float3 scatter_pos = observer_pos + view_dir * s;
        float r = len3(scatter_pos);

        if (r > toa_radius || r < surface_radius) continue;

        int sidx = shell_index(atm, r);
        if (sidx < 0) continue;

        ShellOptics op = read_optics(atm, uint(sidx), wl_idx);
        float beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30f) {
            tau_obs += op.extinction * ds;
            continue;
        }

        float tau_obs_mid = tau_obs + op.extinction * ds * 0.5f;
        float t_obs = exp(-tau_obs_mid);
        if (t_obs < 1e-30f) break;

        // Order 1: deterministic NEE
        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);
        float cos_theta_1 = dot3(sun_dir, -view_dir);
        float phase_1 = mixed_phase(cos_theta_1, op);
        float di_single = beta_scat * phase_1 / (4.0f * PI) * t_sun * t_obs * ds;
        radiance += di_single;

        // Orders 2+: MC secondary chains
        if (secondary_rays > 0) {
            float mc_sum = 0.0f;
            for (uint ray = 0; ray < secondary_rays; ray++) {
                mc_sum += trace_secondary_chain(atm, scatter_pos, sun_dir, wl_idx, op, rng);
            }
            float mc_avg = mc_sum / float(secondary_rays);
            float di_multi = beta_scat * t_obs * ds * mc_avg;
            radiance += di_multi;
        }

        tau_obs += op.extinction * ds;
    }

    output[tid] = radiance;
}

// ============================================================================
// Kernel 4: garstang_zenith
//
// One thread per light source. Computes contribution to zenith brightness.
// Output: brightness[source_idx]
//
// Sources buffer layout: PackedLightSource[N]
// Each source: 8 f32 (2 x vec4)
//   vec4(distance_m, zenith_angle_rad, radiance_wm2sr, spectrum_type)
//   vec4(height_m, pad, pad, pad)
//
// Config buffer:
//   vec4(observer_elevation, aod_550, uplight_fraction, ground_reflectance)
//   vec4(wavelength_nm, altitude_steps, max_altitude, num_sources)
// ============================================================================

kernel void garstang_zenith(
    device const float* sources    [[buffer(0)]],
    device const float* config     [[buffer(1)]],
    device float*       output     [[buffer(2)]],
    uint                tid        [[thread_position_in_grid]]
) {
    // Read config
    float observer_elevation = config[0];
    float aod_550            = config[1];
    float uplight_fraction   = config[2];
    float ground_reflectance = config[3];
    float wavelength_nm      = config[4];
    uint  altitude_steps     = uint(config[5]);
    float max_altitude       = config[6];
    uint  num_sources        = uint(config[7]);

    if (tid >= num_sources) return;

    // Read this source (8 f32 per source)
    uint base = tid * 8;
    float distance_m   = sources[base + 0];
    // sources[base + 1] = zenith_angle_rad (unused for zenith calc)
    float source_rad   = sources[base + 2];
    // sources[base + 3] = spectrum_type (unused)
    // sources[base + 4] = height_m (unused for zenith calc)

    if (distance_m < 1.0f) {
        output[tid] = 0.0f;
        return;
    }

    // Rayleigh optical depth at this wavelength: lambda^-4 scaling
    float wl_ratio = 550.0f / wavelength_nm;
    float rayleigh_tau = TAU_RAYLEIGH_550 * wl_ratio * wl_ratio * wl_ratio * wl_ratio;

    // Aerosol optical depth: Angstrom exponent ~1.3
    float aerosol_tau = aod_550 * pow(wl_ratio, 1.3f);

    float effective_up = uplight_fraction + ground_reflectance * 0.5f;
    float source_intensity = source_rad * effective_up;

    float dh = max_altitude / float(altitude_steps);
    float integral = 0.0f;
    float d = distance_m;

    for (uint step = 0; step < altitude_steps; step++) {
        float h = (float(step) + 0.5f) * dh;
        if (h < observer_elevation) continue;

        float r_src_to_scat = sqrt(d * d + h * h);
        float theta_scatter = PI - atan(d / max(h, 1e-6f));

        // Scattering coefficients at this altitude
        float n_rayleigh = rayleigh_tau / H_RAYLEIGH * exp(-h / H_RAYLEIGH);
        float n_aerosol  = aerosol_tau  / H_AEROSOL  * exp(-h / H_AEROSOL);
        float sigma_total = n_rayleigh + n_aerosol;

        // Phase functions (Garstang uses angle-based, not cos-based)
        float cos_scatter = cos(theta_scatter);
        float p_rayleigh = 3.0f / (16.0f * PI) * (1.0f + cos_scatter * cos_scatter);
        float p_mie = 0.0f;
        {
            float g = 0.7f;
            float g2 = g * g;
            float denom = 1.0f + g2 - 2.0f * g * cos_scatter;
            p_mie = (1.0f - g2) / (4.0f * PI * denom * sqrt(denom));
        }

        float f_rayleigh = (sigma_total > 0.0f) ? (n_rayleigh / sigma_total) : 0.5f;
        float p_avg = f_rayleigh * p_rayleigh + (1.0f - f_rayleigh) * p_mie;

        // Slant optical depth from source to scatter point
        float path_len = r_src_to_scat;
        float tau_slant = 0.0f;
        if (h > 1.0f) {
            float n0_r = rayleigh_tau / H_RAYLEIGH;
            tau_slant += n0_r * path_len * H_RAYLEIGH / h * (1.0f - exp(-h / H_RAYLEIGH));
            float n0_a = aerosol_tau / H_AEROSOL;
            tau_slant += n0_a * path_len * H_AEROSOL / h * (1.0f - exp(-h / H_AEROSOL));
        } else {
            tau_slant = (rayleigh_tau / H_RAYLEIGH + aerosol_tau / H_AEROSOL) * path_len;
        }

        // Vertical optical depth from scatter point to observer
        float tau_vert = rayleigh_tau * (exp(-observer_elevation / H_RAYLEIGH) - exp(-h / H_RAYLEIGH))
                       + aerosol_tau * (exp(-observer_elevation / H_AEROSOL)  - exp(-h / H_AEROSOL));

        float extinction = exp(-tau_slant - tau_vert);
        float r2 = r_src_to_scat * r_src_to_scat;

        float di = source_intensity / (4.0f * PI * r2) * sigma_total * p_avg * extinction * dh;
        integral += di;
    }

    output[tid] = integral;
}
