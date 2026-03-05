// Twilight MCRT - CUDA compute kernels
//
// Four compute kernels for GPU-accelerated twilight radiative transfer:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_scatter            - LOS + secondary MC chains
//   4. garstang_zenith           - Light pollution skyglow
//
// Compiled at runtime via nvrtc (CUDA C -> PTX -> SASS JIT).
// Buffer layout matches crates/twilight-gpu/src/buffers.rs exactly.
// All physics ported from twilight-core (f64) to f32 GPU precision.
// Uses xorshift64 RNG (CUDA supports native unsigned long long).

// ============================================================================
// Constants
// ============================================================================

__device__ constexpr float PI = 3.14159265358979323846f;
__device__ constexpr float EARTH_RADIUS_M = 6371000.0f;
__device__ constexpr float TOA_ALTITUDE_M = 100000.0f;

__device__ constexpr unsigned int MAX_WAVELENGTHS = 64;
__device__ constexpr unsigned int MAX_LOS_STEPS = 200;
__device__ constexpr unsigned int MAX_SCATTERS = 100;
__device__ constexpr unsigned int HYBRID_LOS_STEPS = 200;
__device__ constexpr unsigned int HYBRID_MAX_BOUNCES = 50;

__device__ constexpr float RUSSIAN_ROULETTE_WEIGHT = 0.01f;
__device__ constexpr float RUSSIAN_ROULETTE_SURVIVE = 0.1f;

// Atmosphere buffer offsets (must match buffers.rs atm_offsets)
__device__ constexpr unsigned int ATM_NUM_SHELLS      = 2;
__device__ constexpr unsigned int ATM_NUM_WAVELENGTHS = 3;
__device__ constexpr unsigned int ATM_SHELLS_START    = 4;
__device__ constexpr unsigned int ATM_SHELL_STRIDE    = 4;
__device__ constexpr unsigned int ATM_OPTICS_START    = 260;   // 4 + 4*64
__device__ constexpr unsigned int ATM_OPTICS_STRIDE   = 4;
__device__ constexpr unsigned int ATM_ALBEDO_START    = 16708; // 16644 + 64

// Garstang constants
__device__ constexpr float H_RAYLEIGH = 8500.0f;
__device__ constexpr float H_AEROSOL  = 1500.0f;
__device__ constexpr float TAU_RAYLEIGH_550 = 0.0962f;

// ============================================================================
// Vector helpers (float3)
// ============================================================================

__device__ float len3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize3(float3 v) {
    float inv = rsqrtf(dot3(v, v));
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ float3 operator*(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ float3 operator*(float s, float3 v) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

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

__device__ unsigned int atm_num_shells(const float* atm) {
    return (unsigned int)atm[ATM_NUM_SHELLS];
}

__device__ unsigned int atm_num_wavelengths(const float* atm) {
    return (unsigned int)atm[ATM_NUM_WAVELENGTHS];
}

__device__ ShellGeom read_shell(const float* atm, unsigned int shell_idx) {
    unsigned int base = ATM_SHELLS_START + shell_idx * ATM_SHELL_STRIDE;
    ShellGeom s;
    s.r_inner      = atm[base + 0];
    s.r_outer      = atm[base + 1];
    s.altitude_mid = atm[base + 2];
    s.thickness    = atm[base + 3];
    return s;
}

__device__ ShellOptics read_optics(const float* atm, unsigned int shell_idx, unsigned int wl_idx) {
    unsigned int idx = shell_idx * MAX_WAVELENGTHS + wl_idx;
    unsigned int base = ATM_OPTICS_START + idx * ATM_OPTICS_STRIDE;
    ShellOptics op;
    op.extinction        = atm[base + 0];
    op.ssa               = atm[base + 1];
    op.asymmetry         = atm[base + 2];
    op.rayleigh_fraction = atm[base + 3];
    return op;
}

__device__ float read_albedo(const float* atm, unsigned int wl_idx) {
    return atm[ATM_ALBEDO_START + wl_idx];
}

// Dispatch params: 4 x vec4 (16 floats)
__device__ float3 read_observer(const float* p) {
    return make_float3(p[0], p[1], p[2]);
}
__device__ float3 read_view_dir(const float* p) {
    return make_float3(p[4], p[5], p[6]);
}
__device__ float3 read_sun_dir(const float* p) {
    return make_float3(p[8], p[9], p[10]);
}
__device__ unsigned int read_photons_per_wl(const float* p) {
    return __float_as_uint(p[12]);
}
__device__ unsigned int read_secondary_rays(const float* p) {
    return __float_as_uint(p[13]);
}
__device__ unsigned long long read_rng_seed(const float* p) {
    unsigned int lo = __float_as_uint(p[14]);
    unsigned int hi = __float_as_uint(p[15]);
    return (unsigned long long)lo | ((unsigned long long)hi << 32);
}

// ============================================================================
// xorshift64 RNG (CUDA supports native unsigned long long)
// ============================================================================

__device__ float xorshift_f32(unsigned long long &state) {
    unsigned long long x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return (float)(x >> 40) * (1.0f / (float)(1ull << 24));
}

// ============================================================================
// Ray-sphere intersection
// ============================================================================

struct RaySphereHit {
    float t_near;
    float t_far;
    bool hit;
};

__device__ RaySphereHit ray_sphere_intersect(float3 origin, float3 dir, float radius) {
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

    float sqrt_disc = sqrtf(disc);
    float inv_2a = 0.5f / a;
    result.t_near = (-b - sqrt_disc) * inv_2a;
    result.t_far  = (-b + sqrt_disc) * inv_2a;
    result.hit = true;
    return result;
}

// ============================================================================
// Shell index lookup
// ============================================================================

__device__ int shell_index(const float* atm, float r) {
    unsigned int ns = atm_num_shells(atm);
    for (unsigned int s = 0; s < ns; s++) {
        ShellGeom sh = read_shell(atm, s);
        if (r >= sh.r_inner && r < sh.r_outer) {
            return (int)s;
        }
    }
    if (ns > 0) {
        ShellGeom last = read_shell(atm, ns - 1);
        if (r >= last.r_inner && r <= last.r_outer + 1.0f) {
            return (int)(ns - 1);
        }
    }
    return -1;
}

// ============================================================================
// Phase functions
// ============================================================================

__device__ float rayleigh_phase(float cos_theta) {
    return 0.75f * (1.0f + cos_theta * cos_theta);
}

__device__ float henyey_greenstein_phase(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    return (1.0f - g2) / (denom * sqrtf(denom));
}

__device__ float mixed_phase(float cos_theta, ShellOptics op) {
    if (op.rayleigh_fraction > 0.99f) {
        return rayleigh_phase(cos_theta);
    }
    return op.rayleigh_fraction * rayleigh_phase(cos_theta)
         + (1.0f - op.rayleigh_fraction) * henyey_greenstein_phase(cos_theta, op.asymmetry);
}

// ============================================================================
// Ray path through shell (analytical)
// ============================================================================

__device__ float ray_path_through_shell(float3 origin, float3 dir,
                                        float r_inner, float r_outer, float t_max) {
    RaySphereHit outer = ray_sphere_intersect(origin, dir, r_outer);
    if (!outer.hit) return 0.0f;

    float t0_outer = fmaxf(outer.t_near, 0.0f);
    float t1_outer = fminf(outer.t_far, t_max);
    if (t1_outer <= t0_outer + 1e-6f) return 0.0f;

    RaySphereHit inner = ray_sphere_intersect(origin, dir, r_inner);
    if (!inner.hit) {
        return t1_outer - t0_outer;
    }

    float t0_inner = fmaxf(inner.t_near, 0.0f);
    float t1_inner = fminf(inner.t_far, t_max);
    if (t1_inner <= t0_inner + 1e-6f) {
        return t1_outer - t0_outer;
    }

    float total = 0.0f;
    float seg1_end = fminf(t1_outer, t0_inner);
    if (seg1_end > t0_outer) total += seg1_end - t0_outer;
    float seg2_start = fmaxf(t0_outer, t1_inner);
    if (t1_outer > seg2_start) total += t1_outer - seg2_start;

    return total;
}

// ============================================================================
// Shadow ray transmittance
// ============================================================================

__device__ float shadow_ray_transmittance(const float* atm, float3 start_pos,
                                          float3 sun_dir, unsigned int wl_idx) {
    unsigned int ns = atm_num_shells(atm);
    float toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    float surface_radius = EARTH_RADIUS_M;

    RaySphereHit toa_hit = ray_sphere_intersect(start_pos, sun_dir, toa_radius);
    if (!toa_hit.hit || toa_hit.t_far <= 0.0f) return 0.0f;
    float ray_max = toa_hit.t_far;

    RaySphereHit ground_hit = ray_sphere_intersect(start_pos, sun_dir, surface_radius);
    if (ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < ray_max) {
        return 0.0f;
    }

    float tau = 0.0f;
    for (unsigned int s = 0; s < ns; s++) {
        ShellGeom sh = read_shell(atm, s);
        float path = ray_path_through_shell(start_pos, sun_dir, sh.r_inner, sh.r_outer, ray_max);
        if (path > 0.0f) {
            ShellOptics op = read_optics(atm, s, wl_idx);
            tau += op.extinction * path;
            if (tau > 50.0f) return 0.0f;
        }
    }
    return expf(-tau);
}

// ============================================================================
// Sampling functions
// ============================================================================

__device__ float sample_rayleigh_analytic(float xi) {
    float q = 8.0f * xi - 4.0f;
    float disc = q * q * 0.25f + 1.0f;
    float sqrt_disc = sqrtf(disc);
    float a_val = -q * 0.5f + sqrt_disc;
    float b_val = -q * 0.5f - sqrt_disc;
    float u = (a_val >= 0.0f) ? cbrtf(a_val) : -cbrtf(-a_val);
    float v = (b_val >= 0.0f) ? cbrtf(b_val) : -cbrtf(-b_val);
    float mu = u + v;
    return fminf(fmaxf(mu, -1.0f), 1.0f);
}

__device__ float sample_henyey_greenstein(float xi, float g) {
    if (fabsf(g) < 1e-6f) {
        return 2.0f * xi - 1.0f;
    }
    float g2 = g * g;
    float s = (1.0f - g2) / (1.0f - g + 2.0f * g * xi);
    float mu = (1.0f + g2 - s * s) / (2.0f * g);
    return fminf(fmaxf(mu, -1.0f), 1.0f);
}

__device__ float3 scatter_direction(float3 dir, float cos_theta, float phi) {
    float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    float3 w = dir;
    float3 up = (fabsf(w.z) < 0.9f) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 u_vec = normalize3(cross3(w, up));
    float3 v_vec = cross3(w, u_vec);

    float3 new_dir = sin_theta * cos_phi * u_vec
                   + sin_theta * sin_phi * v_vec
                   + cos_theta * w;
    return normalize3(new_dir);
}

__device__ float3 sample_hemisphere(float3 normal, unsigned long long &rng) {
    float xi1 = xorshift_f32(rng);
    float xi2 = xorshift_f32(rng);
    float cos_theta = sqrtf(xi1);
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

__device__ ShellBoundary next_shell_boundary(float3 pos, float3 dir, float r_inner, float r_outer) {
    ShellBoundary result;
    result.found = false;
    result.dist = 1e30f;
    result.is_outward = true;

    RaySphereHit outer = ray_sphere_intersect(pos, dir, r_outer);
    if (outer.hit) {
        if (outer.t_near > 1e-10f) {
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
// Args: atm (atmosphere buffer), params (dispatch params), output (radiance per wl)
// ============================================================================

extern "C" __global__
void single_scatter_spectrum(const float* atm, const float* params, float* output,
                             unsigned int num_threads) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_wl = atm_num_wavelengths(atm);
    if (tid >= num_wl || tid >= num_threads) return;

    unsigned int wl_idx = tid;
    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);

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

    unsigned int num_steps = min(MAX_LOS_STEPS, (unsigned int)(los_end / 500.0f) + 20u);
    float ds = los_end / (float)num_steps;

    float radiance = 0.0f;
    float tau_obs = 0.0f;
    float cos_theta = dot3(sun_dir, -view_dir);

    for (unsigned int step = 0; step < num_steps; step++) {
        float s = ((float)step + 0.5f) * ds;
        float3 scatter_pos = observer_pos + view_dir * s;
        float r = len3(scatter_pos);

        if (r > toa_radius || r < surface_radius) continue;

        int sidx = shell_index(atm, r);
        if (sidx < 0) continue;

        ShellOptics op = read_optics(atm, (unsigned int)sidx, wl_idx);
        float beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30f) {
            tau_obs += op.extinction * ds;
            continue;
        }

        float tau_obs_mid = tau_obs + op.extinction * ds * 0.5f;
        float t_obs = expf(-tau_obs_mid);
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

    // Ground reflection
    if (hits_ground) {
        float albedo = read_albedo(atm, wl_idx);
        if (albedo > 1e-10f) {
            float3 ground_pos = observer_pos + view_dir * los_end;
            float3 ground_normal = normalize3(ground_pos);
            float cos_sun_incidence = dot3(sun_dir, ground_normal);

            if (cos_sun_incidence > 0.0f) {
                float t_sun_ground = shadow_ray_transmittance(atm, ground_pos, sun_dir, wl_idx);
                float t_obs_ground = expf(-tau_obs);
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
// tid = wl_idx * photons_per_wl + photon_idx
// ============================================================================

extern "C" __global__
void mcrt_trace_photon(const float* atm, const float* params, float* output,
                       unsigned int num_threads) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    unsigned int num_wl = atm_num_wavelengths(atm);
    unsigned int photons_per_wl = read_photons_per_wl(params);
    unsigned int total = num_wl * photons_per_wl;
    if (tid >= total) return;

    unsigned int wl_idx = tid / photons_per_wl;
    unsigned int photon_idx = tid % photons_per_wl;

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);

    // Unique seed per (wavelength, photon) pair
    unsigned long long base_seed = read_rng_seed(params);
    unsigned long long rng = base_seed + (unsigned long long)wl_idx;
    rng *= 6364136223846793005ull;
    rng += (unsigned long long)photon_idx;
    rng *= 2862933555777941757ull;
    rng += 1ull;

    float surface_radius = EARTH_RADIUS_M;

    float3 pos = observer_pos;
    float3 dir = view_dir;
    float weight = 1.0f;
    float result_weight = 0.0f;

    for (unsigned int bounce = 0; bounce < MAX_SCATTERS; bounce++) {
        float r = len3(pos);
        int sidx = shell_index(atm, r);
        if (sidx < 0) break;

        ShellGeom sh = read_shell(atm, (unsigned int)sidx);
        ShellOptics op = read_optics(atm, (unsigned int)sidx, wl_idx);

        if (op.extinction < 1e-20f) {
            ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) break;
            pos = pos + dir * (bnd.dist + 1e-3f);
            continue;
        }

        float xi = xorshift_f32(rng);
        float free_path = -logf(1.0f - xi + 1e-30f) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            pos = pos + dir * (bnd.dist + 1e-3f);
            if (!bnd.is_outward && len3(pos) <= surface_radius + 1.0f) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                float3 normal = normalize3(pos);
                dir = sample_hemisphere(normal, rng);
            }
            continue;
        }

        pos = pos + dir * free_path;

        // NEE
        float t_sun = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun > 1e-30f) {
            float cos_angle = dot3(sun_dir, -dir);
            float phase = mixed_phase(cos_angle, op);
            result_weight += weight * t_sun * phase / (4.0f * PI);
        }

        weight *= op.ssa;

        if (weight < RUSSIAN_ROULETTE_WEIGHT) {
            float xi_rr = xorshift_f32(rng);
            if (xi_rr > RUSSIAN_ROULETTE_SURVIVE) break;
            weight /= RUSSIAN_ROULETTE_SURVIVE;
        }

        float cos_theta_s;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta_s = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_s = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        dir = scatter_direction(dir, cos_theta_s, phi);
    }

    output[tid] = result_weight;
}

// ============================================================================
// Secondary chain tracer (used by hybrid_scatter)
// ============================================================================

__device__ float trace_secondary_chain(const float* atm, float3 start_pos,
                                       float3 sun_dir, unsigned int wl_idx,
                                       ShellOptics start_optics, unsigned long long &rng) {
    float3 local_up = normalize3(start_pos);
    float surface_radius = EARTH_RADIUS_M;

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

    for (unsigned int bounce = 0; bounce < HYBRID_MAX_BOUNCES; bounce++) {
        float r = len3(pos);
        int sidx = shell_index(atm, r);
        if (sidx < 0) break;

        ShellGeom sh = read_shell(atm, (unsigned int)sidx);
        ShellOptics op = read_optics(atm, (unsigned int)sidx, wl_idx);

        if (op.extinction < 1e-20f) {
            ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) break;
            pos = pos + current_dir * (bnd.dist + 1e-3f);
            continue;
        }

        float xi = xorshift_f32(rng);
        float free_path = -logf(1.0f - xi + 1e-30f) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            pos = pos + current_dir * (bnd.dist + 1e-3f);
            if (!bnd.is_outward && len3(pos) <= surface_radius + 1.0f) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                if (weight < 1e-30f) break;
                float3 normal = normalize3(pos);
                current_dir = sample_hemisphere(normal, rng);
            }
            continue;
        }

        pos = pos + current_dir * free_path;

        float t_sun_sec = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun_sec > 1e-30f) {
            float cos_angle = dot3(sun_dir, -current_dir);
            float phase = mixed_phase(cos_angle, op);
            total_contribution += weight * t_sun_sec * phase / (4.0f * PI);
        }

        weight *= op.ssa;

        if (weight < RUSSIAN_ROULETTE_WEIGHT) {
            float xi_rr = xorshift_f32(rng);
            if (xi_rr > RUSSIAN_ROULETTE_SURVIVE) break;
            weight /= RUSSIAN_ROULETTE_SURVIVE;
        }

        float cos_theta_s;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta_s = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_s = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        current_dir = scatter_direction(current_dir, cos_theta_s, phi);
    }

    return total_contribution;
}

// ============================================================================
// Kernel 3: hybrid_scatter
//
// One thread per wavelength. LOS integration + secondary MC chains.
// ============================================================================

extern "C" __global__
void hybrid_scatter(const float* atm, const float* params, float* output,
                    unsigned int num_threads) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_wl = atm_num_wavelengths(atm);
    if (tid >= num_wl || tid >= num_threads) return;

    unsigned int wl_idx = tid;
    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);
    unsigned int secondary_rays_count = read_secondary_rays(params);

    unsigned long long rng = read_rng_seed(params) + (unsigned long long)wl_idx;
    rng *= 6364136223846793005ull;
    rng += 1ull;

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

    unsigned int num_steps = min(HYBRID_LOS_STEPS, (unsigned int)(los_end / 500.0f) + 20u);
    float ds = los_end / (float)num_steps;

    float radiance = 0.0f;
    float tau_obs = 0.0f;

    for (unsigned int step = 0; step < num_steps; step++) {
        float s = ((float)step + 0.5f) * ds;
        float3 scatter_pos = observer_pos + view_dir * s;
        float r = len3(scatter_pos);

        if (r > toa_radius || r < surface_radius) continue;

        int sidx = shell_index(atm, r);
        if (sidx < 0) continue;

        ShellOptics op = read_optics(atm, (unsigned int)sidx, wl_idx);
        float beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30f) {
            tau_obs += op.extinction * ds;
            continue;
        }

        float tau_obs_mid = tau_obs + op.extinction * ds * 0.5f;
        float t_obs = expf(-tau_obs_mid);
        if (t_obs < 1e-30f) break;

        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);
        float cos_theta_1 = dot3(sun_dir, -view_dir);
        float phase_1 = mixed_phase(cos_theta_1, op);
        float di_single = beta_scat * phase_1 / (4.0f * PI) * t_sun * t_obs * ds;
        radiance += di_single;

        if (secondary_rays_count > 0) {
            float mc_sum = 0.0f;
            for (unsigned int ray = 0; ray < secondary_rays_count; ray++) {
                mc_sum += trace_secondary_chain(atm, scatter_pos, sun_dir, wl_idx, op, rng);
            }
            float mc_avg = mc_sum / (float)secondary_rays_count;
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
// One thread per light source. Uses separate buffer layout.
// Sources buffer: 8 floats per source
// Config buffer: 8 floats
// ============================================================================

extern "C" __global__
void garstang_zenith(const float* sources, const float* config, float* output,
                     unsigned int num_threads) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float observer_elevation = config[0];
    float aod_550            = config[1];
    float uplight_fraction   = config[2];
    float ground_reflectance = config[3];
    float wavelength_nm      = config[4];
    unsigned int altitude_steps = (unsigned int)config[5];
    float max_altitude       = config[6];
    unsigned int num_sources = (unsigned int)config[7];

    if (tid >= num_sources || tid >= num_threads) return;

    unsigned int base = tid * 8;
    float distance_m = sources[base + 0];
    float source_rad = sources[base + 2];

    if (distance_m < 1.0f) {
        output[tid] = 0.0f;
        return;
    }

    float wl_ratio = 550.0f / wavelength_nm;
    float rayleigh_tau = TAU_RAYLEIGH_550 * wl_ratio * wl_ratio * wl_ratio * wl_ratio;
    float aerosol_tau = aod_550 * powf(wl_ratio, 1.3f);

    float effective_up = uplight_fraction + ground_reflectance * 0.5f;
    float source_intensity = source_rad * effective_up;

    float dh = max_altitude / (float)altitude_steps;
    float integral = 0.0f;
    float d = distance_m;

    for (unsigned int step = 0; step < altitude_steps; step++) {
        float h = ((float)step + 0.5f) * dh;
        if (h < observer_elevation) continue;

        float r_src_to_scat = sqrtf(d * d + h * h);
        float theta_scatter = PI - atan2f(d, fmaxf(h, 1e-6f));

        float n_rayleigh = rayleigh_tau / H_RAYLEIGH * expf(-h / H_RAYLEIGH);
        float n_aerosol  = aerosol_tau  / H_AEROSOL  * expf(-h / H_AEROSOL);
        float sigma_total = n_rayleigh + n_aerosol;

        float cos_scatter = cosf(theta_scatter);
        float p_rayleigh = 3.0f / (16.0f * PI) * (1.0f + cos_scatter * cos_scatter);
        float g = 0.7f;
        float g2 = g * g;
        float denom = 1.0f + g2 - 2.0f * g * cos_scatter;
        float p_mie = (1.0f - g2) / (4.0f * PI * denom * sqrtf(denom));

        float f_rayleigh = (sigma_total > 0.0f) ? (n_rayleigh / sigma_total) : 0.5f;
        float p_avg = f_rayleigh * p_rayleigh + (1.0f - f_rayleigh) * p_mie;

        float path_len = r_src_to_scat;
        float tau_slant = 0.0f;
        if (h > 1.0f) {
            float n0_r = rayleigh_tau / H_RAYLEIGH;
            tau_slant += n0_r * path_len * H_RAYLEIGH / h * (1.0f - expf(-h / H_RAYLEIGH));
            float n0_a = aerosol_tau / H_AEROSOL;
            tau_slant += n0_a * path_len * H_AEROSOL / h * (1.0f - expf(-h / H_AEROSOL));
        } else {
            tau_slant = (rayleigh_tau / H_RAYLEIGH + aerosol_tau / H_AEROSOL) * path_len;
        }

        float tau_vert = rayleigh_tau * (expf(-observer_elevation / H_RAYLEIGH) - expf(-h / H_RAYLEIGH))
                       + aerosol_tau * (expf(-observer_elevation / H_AEROSOL)  - expf(-h / H_AEROSOL));

        float extinction = expf(-tau_slant - tau_vert);
        float r2 = r_src_to_scat * r_src_to_scat;

        float di = source_intensity / (4.0f * PI * r2) * sigma_total * p_avg * extinction * dh;
        integral += di;
    }

    output[tid] = integral;
}
