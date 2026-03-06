// Twilight MCRT - CUDA compute kernels (v2)
//
// Four compute kernels for GPU-accelerated twilight radiative transfer:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_scatter            - LOS + secondary MC chains (reparallelized)
//   4. garstang_zenith           - Light pollution skyglow
//
// Compiled at runtime via nvrtc (CUDA C -> PTX -> SASS JIT).
// Buffer layout matches crates/twilight-gpu/src/buffers.rs (v2) exactly.
// All physics ported from twilight-core (f64) to f32 GPU precision.
// Uses xorshift64 RNG (CUDA supports native unsigned long long).
//
// Key changes from v1:
//   - Binary search O(log N) shell lookup
//   - Shell-by-shell shadow ray with Snell's law refraction
//   - Radial 2m boundary nudge (not along ray direction)
//   - Kahan compensated summation for optical depth and radiance
//   - Hybrid kernel: 1 block per wavelength with warp shuffle reduction

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
__device__ constexpr unsigned int HYBRID_BLOCK_SIZE = 256;
__device__ constexpr unsigned int WARP_SIZE = 32;
__device__ constexpr unsigned int NUM_WARPS = HYBRID_BLOCK_SIZE / WARP_SIZE; // 8

// Atmosphere buffer offsets (must match buffers.rs atm_offsets)
__device__ constexpr unsigned int ATM_NUM_SHELLS            = 2;
__device__ constexpr unsigned int ATM_NUM_WAVELENGTHS       = 3;
__device__ constexpr unsigned int ATM_SHELLS_START          = 4;
__device__ constexpr unsigned int ATM_SHELL_STRIDE          = 4;
__device__ constexpr unsigned int ATM_OPTICS_START          = 260;   // 4 + 4*64
__device__ constexpr unsigned int ATM_OPTICS_STRIDE         = 4;
__device__ constexpr unsigned int ATM_ALBEDO_START          = 16708; // 16644 + 64
__device__ constexpr unsigned int ATM_REFRACTIVE_INDEX_START = 16772; // 16708 + 64 (v2)

// Garstang constants
__device__ constexpr float H_RAYLEIGH = 8500.0f;
__device__ constexpr float H_AEROSOL  = 1500.0f;
__device__ constexpr float TAU_RAYLEIGH_550 = 0.0962f;

// Boundary nudge distance (meters)
__device__ constexpr float BOUNDARY_NUDGE_M = 2.0f;

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
    float inv = rsqrtf(dot3(v, v) + 1e-30f);
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
// KBN (Kahan-Babuska-Neumaier) compensated summation
// ============================================================================

struct KahanAccum {
    float sum;
    float comp;

    __device__ KahanAccum() : sum(0.0f), comp(0.0f) {}

    __device__ void add(float value) {
        float t = sum + value;
        if (fabsf(sum) >= fabsf(value)) {
            comp += (sum - t) + value;
        } else {
            comp += (value - t) + sum;
        }
        sum = t;
    }

    __device__ float result() const { return sum + comp; }
};

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

__device__ float read_refractive_index(const float* atm, unsigned int shell_idx) {
    return atm[ATM_REFRACTIVE_INDEX_START + shell_idx];
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
// xorshift64 RNG
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
// Error-free transformations (DS arithmetic via FMA)
// ============================================================================

struct DS {
    float hi;
    float lo;
};

__device__ DS two_product(float a, float b) {
    float p = a * b;
    float e = __fmaf_rn(a, b, -p);
    DS result;
    result.hi = p;
    result.lo = e;
    return result;
}

__device__ DS two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    DS result;
    result.hi = s;
    result.lo = e;
    return result;
}

__device__ DS ds_add(DS x, DS y) {
    DS s = two_sum(x.hi, y.hi);
    s.lo += x.lo + y.lo;
    DS r = two_sum(s.hi, s.lo);
    return r;
}

__device__ DS ds_sub(DS x, DS y) {
    DS neg_y;
    neg_y.hi = -y.hi;
    neg_y.lo = -y.lo;
    return ds_add(x, neg_y);
}

// ============================================================================
// Ray-sphere intersection (DS discriminant + stable quadratic)
// ============================================================================

struct RaySphereHit {
    float t_near;
    float t_far;
    bool hit;
};

__device__ RaySphereHit ray_sphere_intersect(float3 origin, float3 dir, float radius) {
    float a = dot3(dir, dir);
    float b_half = dot3(origin, dir);
    float r_pos = len3(origin);
    float c = (r_pos - radius) * (r_pos + radius);

    // DS discriminant: b_half^2 - a*c with ~14-digit precision
    DS b2 = two_product(b_half, b_half);
    DS ac = two_product(a, c);
    DS disc_ds = ds_sub(b2, ac);
    float disc = disc_ds.hi + disc_ds.lo;

    RaySphereHit result;
    if (disc < 0.0f) {
        result.hit = false;
        result.t_near = 0.0f;
        result.t_far = 0.0f;
        return result;
    }

    float sqrt_disc = sqrtf(fmaxf(disc, 0.0f));

    // Stable quadratic using copysignf
    float q = -(b_half + copysignf(sqrt_disc, b_half));

    float t1, t2;
    if (fabsf(q) > 1e-30f) {
        t1 = q / a;
        t2 = c / q;
    } else {
        float inv_a = 1.0f / a;
        t1 = (-b_half - sqrt_disc) * inv_a;
        t2 = (-b_half + sqrt_disc) * inv_a;
    }

    result.t_near = fminf(t1, t2);
    result.t_far  = fmaxf(t1, t2);
    result.hit = true;
    return result;
}

// ============================================================================
// Shell index lookup -- O(log N) binary search
// ============================================================================

__device__ int shell_index_binary(const float* atm, float r) {
    unsigned int ns = atm_num_shells(atm);
    if (ns == 0) return -1;

    float r_inner_first = atm[ATM_SHELLS_START];
    float r_outer_last = atm[ATM_SHELLS_START + (ns - 1) * ATM_SHELL_STRIDE + 1];
    if (r < r_inner_first || r >= r_outer_last) return -1;

    unsigned int lo = 0;
    unsigned int hi = ns;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        float r_inner_mid = atm[ATM_SHELLS_START + mid * ATM_SHELL_STRIDE];
        if (r_inner_mid <= r) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo == 0) ? -1 : (int)(lo - 1);
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
// float4 helpers
// ============================================================================

__device__ float4 make_f4(float x, float y, float z, float w) {
    return make_float4(x, y, z, w);
}

__device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float4 operator*(float4 v, float s) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__device__ float4 operator*(float s, float4 v) {
    return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

// ============================================================================
// Stokes [I,Q,U,V] polarized RT helpers
// ============================================================================

__device__ float rayleigh_P12(float cos_theta) {
    float sin2 = 1.0f - cos_theta * cos_theta;
    return -0.75f * sin2;
}

__device__ float rayleigh_P33(float cos_theta) {
    return 1.5f * cos_theta;
}

__device__ void scattering_plane_rotation(float3 dir_in, float3 dir_out, float3 dir_next,
                                           float &cos2phi, float &sin2phi) {
    float3 n1 = cross3(dir_in, dir_out);
    float3 n2 = cross3(dir_out, dir_next);

    float n1_sq = dot3(n1, n1);
    float n2_sq = dot3(n2, n2);

    if (n1_sq < 1e-20f || n2_sq < 1e-20f) {
        cos2phi = 1.0f;
        sin2phi = 0.0f;
        return;
    }

    float inv_norm = rsqrtf(n1_sq * n2_sq);
    float cos_phi = dot3(n1, n2) * inv_norm;
    float sin_phi = dot3(dir_out, cross3(n1, n2)) * inv_norm;

    cos_phi = fminf(fmaxf(cos_phi, -1.0f), 1.0f);

    cos2phi = 2.0f * cos_phi * cos_phi - 1.0f;
    sin2phi = 2.0f * sin_phi * cos_phi;
}

__device__ void stokes_ABC(float cos_theta, ShellOptics op,
                           float &A, float &B, float &C) {
    float alpha = op.rayleigh_fraction;
    float p11_r = rayleigh_phase(cos_theta);
    float p12_r = rayleigh_P12(cos_theta);
    float p33_r = rayleigh_P33(cos_theta);
    float p11_hg = henyey_greenstein_phase(cos_theta, op.asymmetry);

    A = alpha * p11_r + (1.0f - alpha) * p11_hg;
    B = alpha * p12_r;
    C = alpha * p33_r + (1.0f - alpha) * p11_hg;
}

__device__ float4 scatter_stokes(float A, float B, float C,
                                  float cos2phi, float sin2phi, float4 s_in) {
    float rotQU = cos2phi * s_in.y + sin2phi * s_in.z;
    float4 s_out;
    s_out.x = A * s_in.x + B * rotQU;
    s_out.y = B * s_in.x + A * rotQU;
    s_out.z = C * (cos2phi * s_in.z - sin2phi * s_in.y);
    s_out.w = C * s_in.w;
    return s_out;
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

    const float EPS = 1e-5f;

    RaySphereHit outer = ray_sphere_intersect(pos, dir, r_outer);
    if (outer.hit) {
        if (outer.t_near > EPS) {
            RaySphereHit inner = ray_sphere_intersect(pos, dir, r_inner);
            if (inner.hit && inner.t_near > EPS && inner.t_near < outer.t_near) {
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
        if (outer.t_far > EPS) {
            RaySphereHit inner = ray_sphere_intersect(pos, dir, r_inner);
            if (inner.hit && inner.t_near > EPS && inner.t_near < outer.t_far) {
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
    if (inner.hit && inner.t_near > EPS) {
        result.dist = inner.t_near;
        result.is_outward = false;
        result.found = true;
    }
    return result;
}

// ============================================================================
// Snell's law refraction at spherical shell boundary
// ============================================================================

__device__ float3 refract_at_boundary(float3 dir, float3 boundary_pos, float n_from, float n_to) {
    if (fabsf(n_from - n_to) < 1e-7f) return dir;

    float3 outward = normalize3(boundary_pos);
    float cos_dir_normal = dot3(dir, outward);
    float3 normal = (cos_dir_normal < 0.0f) ? outward : -outward;

    float cos_i = -dot3(dir, normal);
    float eta = n_from / n_to;
    float k = 1.0f - eta * eta * (1.0f - cos_i * cos_i);

    if (k < 0.0f) {
        // Total internal reflection: result is unit by reflection identity.
        return dir + normal * (2.0f * cos_i);
    }

    float cos_t = sqrtf(k);
    float factor = eta * cos_i - cos_t;
    // Snell refraction: result is unit by Snell's law identity.
    return dir * eta + normal * factor;
}

// ============================================================================
// Radial boundary nudge
// ============================================================================

__device__ float3 radial_nudge(float3 boundary_pos, bool is_outward) {
    float bp_r = len3(boundary_pos);
    float3 radial_dir = (bp_r > 1e-10f) ?
        make_float3(boundary_pos.x / bp_r, boundary_pos.y / bp_r, boundary_pos.z / bp_r) :
        make_float3(1.0f, 0.0f, 0.0f);
    float nudge_sign = is_outward ? 1.0f : -1.0f;
    return boundary_pos + radial_dir * (nudge_sign * BOUNDARY_NUDGE_M);
}

// ============================================================================
// Shadow ray transmittance -- shell-by-shell with refraction + Kahan
// ============================================================================

__device__ float shadow_ray_transmittance(const float* atm, float3 start_pos,
                                          float3 sun_dir, unsigned int wl_idx) {
    unsigned int ns = atm_num_shells(atm);
    float surface_radius = atm[ATM_SHELLS_START]; // r_inner of shell 0

    // Umbra cylinder culling (O(1) pre-check)
    float p_proj = dot3(start_pos, sun_dir);
    if (p_proj < 0.0f) {
        float3 cross_ps = cross3(start_pos, sun_dir);
        float perp_dist_sq = dot3(cross_ps, cross_ps);
        if (perp_dist_sq < surface_radius * surface_radius) {
            return 0.0f;
        }
    }

    float3 pos = start_pos;
    float3 dir = sun_dir;

    KahanAccum tau;

    // Find initial shell once (O(log N)), then track directly (O(1) per step).
    int sidx = shell_index_binary(atm, len3(pos));
    if (sidx < 0) return 1.0f;
    unsigned int us = (unsigned int)sidx;

    for (unsigned int iter = 0; iter < 200; iter++) {
        float r_inner = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE];
        float r_outer = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE + 1];

        unsigned int optics_idx = us * MAX_WAVELENGTHS + wl_idx;
        float extinction = atm[ATM_OPTICS_START + optics_idx * ATM_OPTICS_STRIDE];

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) break;

        tau.add(extinction * bnd.dist);

        float3 boundary_pos = pos + dir * bnd.dist;
        // Snap to exact boundary radius to prevent cumulative f32 position drift
        float target_r = bnd.is_outward ? r_outer : r_inner;
        float bp_len = len3(boundary_pos);
        if (bp_len > 0.0f) {
            boundary_pos = boundary_pos * (target_r / bp_len);
        }
        float n_from = read_refractive_index(atm, us);
        unsigned int next_shell = bnd.is_outward ? us + 1 : us - 1;
        float n_to = (next_shell < ns) ? read_refractive_index(atm, next_shell) : 1.0f;

        dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
        pos = radial_nudge(boundary_pos, bnd.is_outward);

        if (!bnd.is_outward && len3(pos) <= surface_radius + 1.0f) {
            return 0.0f;
        }

        // Exited atmosphere
        if (next_shell >= ns) break;
        us = next_shell;

        if (tau.result() > 50.0f) return 0.0f;
    }

    return expf(-tau.result());
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
    float cos_phi, sin_phi;
    sincosf(phi, &sin_phi, &cos_phi);

    float3 w = dir;
    float3 up = (fabsf(w.z) < 0.9f) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 u_vec = normalize3(cross3(w, up));
    float3 v_vec = cross3(w, u_vec);

    // (u_vec, v_vec, w) is orthonormal: result is unit length, no normalize needed.
    float sc = sin_theta * cos_phi;
    float ss = sin_theta * sin_phi;
    return sc * u_vec + ss * v_vec + cos_theta * w;
}

__device__ float3 sample_hemisphere(float3 normal, unsigned long long &rng) {
    float xi1 = xorshift_f32(rng);
    float xi2 = xorshift_f32(rng);
    float cos_theta = sqrtf(xi1);
    float phi = 2.0f * PI * xi2;
    return scatter_direction(normal, cos_theta, phi);
}

// ============================================================================
// Warp-level reduction using __shfl_down_sync
// ============================================================================

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float4 warp_reduce_sum4(float4 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
        val.z += __shfl_down_sync(0xffffffff, val.z, offset);
        val.w += __shfl_down_sync(0xffffffff, val.w, offset);
    }
    return val;
}

// ============================================================================
// Kernel 1: single_scatter_spectrum
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

    KahanAccum radiance;
    KahanAccum tau_obs;
    float cos_theta = dot3(sun_dir, -view_dir);

    for (unsigned int step = 0; step < num_steps; step++) {
        float s = ((float)step + 0.5f) * ds;
        float3 scatter_pos = observer_pos + view_dir * s;
        float r = len3(scatter_pos);

        if (r > toa_radius || r < surface_radius) continue;

        int sidx = shell_index_binary(atm, r);
        if (sidx < 0) continue;

        ShellOptics op = read_optics(atm, (unsigned int)sidx, wl_idx);
        float beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30f) {
            tau_obs.add(op.extinction * ds);
            continue;
        }

        // exp(-(A+B)) = exp(-A)*exp(-B) avoids f32 precision loss when
        // adding a small half-step to a large accumulated tau.
        float t_obs = expf(-tau_obs.result()) * expf(-op.extinction * ds * 0.5f);
        if (t_obs < 1e-30f) break;

        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);
        if (t_sun < 1e-30f) {
            tau_obs.add(op.extinction * ds);
            continue;
        }

        float phase = mixed_phase(cos_theta, op);
        float di = beta_scat * phase / (4.0f * PI) * t_sun * t_obs * ds;
        radiance.add(di);
        tau_obs.add(op.extinction * ds);
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
                float t_obs_ground = expf(-tau_obs.result());
                radiance.add(albedo / PI * cos_sun_incidence * t_sun_ground * t_obs_ground);
            }
        }
    }

    output[tid] = radiance.result();
}

// ============================================================================
// Kernel 2: mcrt_trace_photon
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

    unsigned long long base_seed = read_rng_seed(params);
    unsigned long long rng = base_seed + (unsigned long long)wl_idx;
    rng *= 6364136223846793005ull;
    rng += (unsigned long long)photon_idx;
    rng *= 2862933555777941757ull;
    rng += 1ull;

    float surface_radius = EARTH_RADIUS_M;

    float3 pos = observer_pos;
    float3 dir = view_dir;
    float3 prev_dir = dir;
    float4 stokes = make_f4(1.0f, 0.0f, 0.0f, 0.0f);
    float weight = 1.0f;
    KahanAccum result_weight;

    for (unsigned int bounce = 0; bounce < MAX_SCATTERS; bounce++) {
        float r = len3(pos);
        int sidx = shell_index_binary(atm, r);
        if (sidx < 0) break;

        unsigned int us = (unsigned int)sidx;
        ShellGeom sh = read_shell(atm, us);
        ShellOptics op = read_optics(atm, us, wl_idx);

        if (op.extinction < 1e-20f) {
            ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) break;
            float3 boundary_pos = pos + dir * bnd.dist;
            float n_from = read_refractive_index(atm, us);
            unsigned int next_s = bnd.is_outward ? us + 1 : us - 1;
            float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
            dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        float xi = xorshift_f32(rng);
        float free_path = -logf(1.0f - xi + 1e-30f) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            float3 boundary_pos = pos + dir * bnd.dist;

            // Ground reflection: depolarizes
            if (!bnd.is_outward && len3(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                float3 normal = normalize3(boundary_pos);
                prev_dir = dir;
                dir = sample_hemisphere(normal, rng);
                pos = radial_nudge(boundary_pos, true);
                stokes = make_f4(1.0f, 0.0f, 0.0f, 0.0f);
                continue;
            }

            // Refract and nudge past boundary
            {
                float n_from = read_refractive_index(atm, us);
                unsigned int next_s = bnd.is_outward ? us + 1 : us - 1;
                float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
                dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
            }
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        pos = pos + dir * free_path;

        // NEE: apply Mueller to photon's current Stokes state
        float t_sun = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun > 1e-30f) {
            float cos_angle = dot3(sun_dir, -dir);
            float A_nee, B_nee, C_nee;
            stokes_ABC(cos_angle, op, A_nee, B_nee, C_nee);
            float cos2phi_nee, sin2phi_nee;
            scattering_plane_rotation(prev_dir, dir, -sun_dir, cos2phi_nee, sin2phi_nee);
            float4 nee_stokes = scatter_stokes(A_nee, B_nee, C_nee, cos2phi_nee, sin2phi_nee, stokes);
            result_weight.add(weight * t_sun * nee_stokes.x / (4.0f * PI));
        }

        weight *= op.ssa;

        // Sample new direction and update Stokes state
        float cos_theta_s;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta_s = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_s = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        float3 new_dir = scatter_direction(dir, cos_theta_s, phi);

        float A_s, B_s, C_s;
        stokes_ABC(cos_theta_s, op, A_s, B_s, C_s);
        float cos2phi_s, sin2phi_s;
        scattering_plane_rotation(prev_dir, dir, new_dir, cos2phi_s, sin2phi_s);
        stokes = scatter_stokes(A_s, B_s, C_s, cos2phi_s, sin2phi_s, stokes);
        if (stokes.x > 1e-30f) {
            stokes = stokes * (1.0f / stokes.x);
        } else {
            stokes = make_f4(1.0f, 0.0f, 0.0f, 0.0f);
        }

        prev_dir = dir;
        dir = new_dir;
    }

    output[tid] = result_weight.result();
}

// ============================================================================
// Secondary chain tracer (full Stokes [I,Q,U,V])
// ============================================================================

__device__ float4 trace_secondary_chain(const float* atm, float3 start_pos,
                                        float3 sun_dir, unsigned int wl_idx,
                                        ShellOptics start_optics, float3 prev_dir_in,
                                        unsigned long long &rng) {
    float3 local_up = normalize3(start_pos);
    float surface_radius = EARTH_RADIUS_M;

    float xi_mix = xorshift_f32(rng);
    float3 dir;
    float cos_theta_init;
    if (xi_mix < 0.5f) {
        if (xorshift_f32(rng) < start_optics.rayleigh_fraction) {
            cos_theta_init = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_init = sample_henyey_greenstein(xorshift_f32(rng), start_optics.asymmetry);
        }
        float phi_init = 2.0f * PI * xorshift_f32(rng);
        dir = scatter_direction(sun_dir, cos_theta_init, phi_init);
    } else {
        dir = sample_hemisphere(local_up, rng);
        cos_theta_init = dot3(sun_dir, dir);
    }

    // Initialize Stokes state with first scatter Mueller
    float4 stokes = make_f4(1.0f, 0.0f, 0.0f, 0.0f);
    {
        float A0, B0, C0;
        stokes_ABC(cos_theta_init, start_optics, A0, B0, C0);
        float cos2phi0, sin2phi0;
        scattering_plane_rotation(prev_dir_in, sun_dir, dir, cos2phi0, sin2phi0);
        stokes = scatter_stokes(A0, B0, C0, cos2phi0, sin2phi0, stokes);
        if (stokes.x > 1e-30f) {
            stokes = stokes * (1.0f / stokes.x);
        }
    }

    float3 pos = start_pos;
    float3 current_dir = dir;
    float3 prev_dir = sun_dir;
    float weight = start_optics.ssa;

    KahanAccum total_I, total_Q, total_U, total_V;

    for (unsigned int bounce = 0; bounce < HYBRID_MAX_BOUNCES; bounce++) {
        float r = len3(pos);
        int sidx = shell_index_binary(atm, r);
        if (sidx < 0) break;

        unsigned int us = (unsigned int)sidx;
        ShellGeom sh = read_shell(atm, us);
        ShellOptics op = read_optics(atm, us, wl_idx);

        if (op.extinction < 1e-20f) {
            ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) break;
            float3 boundary_pos = pos + current_dir * bnd.dist;
            float n_from = read_refractive_index(atm, us);
            unsigned int next_s = bnd.is_outward ? us + 1 : us - 1;
            float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
            current_dir = refract_at_boundary(current_dir, boundary_pos, n_from, n_to);
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        float xi = xorshift_f32(rng);
        float free_path = -logf(1.0f - xi + 1e-30f) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            float3 boundary_pos = pos + current_dir * bnd.dist;

            if (!bnd.is_outward && len3(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                if (weight < 1e-30f) break;
                float3 normal = normalize3(boundary_pos);
                prev_dir = current_dir;
                current_dir = sample_hemisphere(normal, rng);
                pos = radial_nudge(boundary_pos, true);
                stokes = make_f4(1.0f, 0.0f, 0.0f, 0.0f);
                continue;
            }

            // Refract and nudge past boundary
            {
                float n_from = read_refractive_index(atm, us);
                unsigned int next_s = bnd.is_outward ? us + 1 : us - 1;
                float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
                current_dir = refract_at_boundary(current_dir, boundary_pos, n_from, n_to);
            }
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        pos = pos + current_dir * free_path;

        // NEE with full Stokes
        float t_sun_sec = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun_sec > 1e-30f) {
            float cos_angle_nee = dot3(sun_dir, -current_dir);
            float A_nee, B_nee, C_nee;
            stokes_ABC(cos_angle_nee, op, A_nee, B_nee, C_nee);

            float cos2phi_nee, sin2phi_nee;
            scattering_plane_rotation(prev_dir, current_dir, -sun_dir, cos2phi_nee, sin2phi_nee);

            float4 nee_stokes = scatter_stokes(A_nee, B_nee, C_nee, cos2phi_nee, sin2phi_nee, stokes);

            float scale = weight * t_sun_sec / (4.0f * PI);
            total_I.add(scale * nee_stokes.x);
            total_Q.add(scale * nee_stokes.y);
            total_U.add(scale * nee_stokes.z);
            total_V.add(scale * nee_stokes.w);
        }

        weight *= op.ssa;
        if (weight < 1e-30f) break;

        float cos_theta_s;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta_s = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta_s = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        float3 new_dir = scatter_direction(current_dir, cos_theta_s, phi);

        float A_s, B_s, C_s;
        stokes_ABC(cos_theta_s, op, A_s, B_s, C_s);
        float cos2phi_s, sin2phi_s;
        scattering_plane_rotation(prev_dir, current_dir, new_dir, cos2phi_s, sin2phi_s);
        stokes = scatter_stokes(A_s, B_s, C_s, cos2phi_s, sin2phi_s, stokes);

        if (stokes.x > 1e-30f) {
            stokes = stokes * (1.0f / stokes.x);
        } else {
            stokes = make_f4(1.0f, 0.0f, 0.0f, 0.0f);
        }

        prev_dir = current_dir;
        current_dir = new_dir;
    }

    return make_f4(total_I.result(), total_Q.result(), total_U.result(), total_V.result());
}

// ============================================================================
// Kernel 3: hybrid_scatter (REPARALLELIZED)
//
// 1 block per wavelength, HYBRID_BLOCK_SIZE threads per block.
// Each thread handles one LOS step + secondary chains.
// Reduction: __shfl_down_sync + __shared__ memory.
//
// Dispatch: num_wavelengths blocks of HYBRID_BLOCK_SIZE threads.
//   blockIdx.x  = wl_idx
//   threadIdx.x = step_idx
// ============================================================================

extern "C" __global__
void hybrid_scatter(const float* atm, const float* params, float* output,
                    unsigned int num_threads) {
    unsigned int wl_idx = blockIdx.x;
    unsigned int step_idx = threadIdx.x;
    unsigned int num_wl = atm_num_wavelengths(atm);
    if (wl_idx >= num_wl) return;

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);
    unsigned int secondary_rays_count = read_secondary_rays(params);

    float toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    float surface_radius = EARTH_RADIUS_M;

    // LOS geometry
    RaySphereHit toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    bool valid_los = toa_hit.hit && toa_hit.t_far > 0.0f;

    unsigned int num_steps = 0;
    float ds = 0.0f;

    if (valid_los) {
        float los_max = toa_hit.t_far;
        RaySphereHit ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
        bool hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < los_max;
        float los_end = hits_ground ? ground_hit.t_near : los_max;
        if (los_end > 0.0f) {
            num_steps = min(HYBRID_LOS_STEPS, (unsigned int)(los_end / 500.0f) + 20u);
            ds = los_end / (float)num_steps;
        }
    }

    // Phase 1: compute ext*ds per step
    __shared__ float shared_ext_ds[HYBRID_BLOCK_SIZE];

    float my_ext_ds = 0.0f;
    float my_beta_scat = 0.0f;
    float3 scatter_pos = make_float3(0.0f, 0.0f, 0.0f);
    int my_sidx = -1;
    ShellOptics my_op;
    my_op.extinction = 0.0f;
    my_op.ssa = 0.0f;
    my_op.asymmetry = 0.0f;
    my_op.rayleigh_fraction = 0.0f;

    if (valid_los && step_idx < num_steps) {
        float s = ((float)step_idx + 0.5f) * ds;
        scatter_pos = observer_pos + view_dir * s;
        float r = len3(scatter_pos);

        if (r <= toa_radius && r >= surface_radius) {
            my_sidx = shell_index_binary(atm, r);
            if (my_sidx >= 0) {
                my_op = read_optics(atm, (unsigned int)my_sidx, wl_idx);
                my_ext_ds = my_op.extinction * ds;
                my_beta_scat = my_op.extinction * my_op.ssa;
            }
        }
    }

    shared_ext_ds[step_idx] = my_ext_ds;
    __syncthreads();

    // Phase 2: compute tau_obs via prefix scan
    float tau_obs = 0.0f;
    for (unsigned int i = 0; i < step_idx && i < num_steps; i++) {
        tau_obs += shared_ext_ds[i];
    }
    // exp(-(A+B)) = exp(-A)*exp(-B) avoids f32 precision loss
    float t_obs = expf(-tau_obs) * expf(-my_ext_ds * 0.5f);

    // Phase 3: per-step Stokes contribution
    float4 contribution = make_f4(0.0f, 0.0f, 0.0f, 0.0f);

    if (valid_los && step_idx < num_steps && my_sidx >= 0
        && my_beta_scat > 1e-30f && t_obs > 1e-30f)
    {
        unsigned long long rng = read_rng_seed(params) + (unsigned long long)wl_idx;
        rng *= 6364136223846793005ull;
        rng += (unsigned long long)step_idx;
        rng *= 2862933555777941757ull;
        rng += 1ull;

        // Order 1: single-scatter NEE (Stokes)
        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);
        if (t_sun > 1e-30f) {
            float cos_theta_1 = dot3(sun_dir, -view_dir);
            float A_1, B_1, C_1;
            stokes_ABC(cos_theta_1, my_op, A_1, B_1, C_1);
            float4 ss_stokes = make_f4(A_1, B_1, 0.0f, 0.0f);
            float scale_1 = my_beta_scat / (4.0f * PI) * t_sun * t_obs * ds;
            contribution = contribution + ss_stokes * scale_1;
        }

        // Orders 2+: MC secondary chains (full Stokes, noise gate REMOVED)
        if (secondary_rays_count > 0) {
            KahanAccum mc_I, mc_Q, mc_U, mc_V;
            for (unsigned int ray = 0; ray < secondary_rays_count; ray++) {
                float4 chain = trace_secondary_chain(atm, scatter_pos, sun_dir, wl_idx,
                                                      my_op, view_dir, rng);
                mc_I.add(chain.x);
                mc_Q.add(chain.y);
                mc_U.add(chain.z);
                mc_V.add(chain.w);
            }
            float inv_rays = 1.0f / (float)secondary_rays_count;
            float4 mc_avg = make_f4(mc_I.result(), mc_Q.result(),
                                     mc_U.result(), mc_V.result()) * inv_rays;
            float scale_m = my_beta_scat * t_obs * ds;
            contribution = contribution + mc_avg * scale_m;
        }
    }

    // Phase 4: two-level Stokes reduction
    float4 warp_total = warp_reduce_sum4(contribution);

    __shared__ float4 shared_sums[NUM_WARPS];
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        shared_sums[warp_id] = warp_total;
    }
    __syncthreads();

    if (step_idx == 0) {
        float4 total = make_f4(0.0f, 0.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i < NUM_WARPS; i++) {
            total = total + shared_sums[i];
        }
        output[wl_idx] = total.x;  // Stokes I = total intensity
    }
}

// ============================================================================
// Kernel 4: garstang_zenith
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
    KahanAccum integral;
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
        integral.add(di);
    }

    output[tid] = integral.result();
}
