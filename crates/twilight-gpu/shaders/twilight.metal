// Twilight MCRT - Metal Shading Language compute kernels (v2)
//
// Four compute kernels for GPU-accelerated twilight radiative transfer:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_scatter            - LOS + secondary MC chains (reparallelized)
//   4. garstang_zenith           - Light pollution skyglow
//
// Buffer layout matches crates/twilight-gpu/src/buffers.rs (v2) exactly.
// All physics ported from twilight-core (f64) to f32 GPU precision.
//
// Key changes from v1:
//   - Binary search O(log N) shell lookup (was O(N) linear scan)
//   - Shell-by-shell shadow ray with Snell's law refraction at boundaries
//   - Radial boundary nudge (2m along Earth normal, not along ray)
//   - Kahan compensated summation for optical depth and radiance
//   - Hybrid kernel: 1 threadgroup per wavelength with SIMD reduction
//     (was 1 thread per wavelength -- 23x slower than CPU)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant float PI = M_PI_F;
constant float INV_4PI = 1.0f / (4.0f * PI); // isotropic phase pdf
// Exact IUGG R1 mean radius, matching twilight-core. Fallback only: all
// kernels derive surface/TOA radii from the packed shells (see
// atm_surface_radius / atm_toa_radius).
constant float EARTH_RADIUS_M = 6371008.7714f;
constant float TOA_ALTITUDE_M = 100000.0f;

constant uint MAX_WAVELENGTHS = 64;
constant uint MAX_LOS_STEPS = 200;
constant uint MAX_SCATTERS = 100;
constant uint HYBRID_LOS_STEPS = 200;
constant uint HYBRID_MAX_BOUNCES = 50;
// v1 kernel (step-parallel) uses 256 threads.
// v2 kernel (ray-parallel) uses 64 threads to stay within Metal's per-
// threadgroup stack limit -- each trace_secondary_chain needs ~2-4 KB
// stack per thread, and 256 * 4 KB = 1 MB would exceed the default.
constant uint HYBRID_THREADGROUP_SIZE = 256;
constant uint HYBRID_V2_THREADGROUP_SIZE = 64;
constant uint SIMD_WIDTH = 32;
constant uint NUM_SIMD_GROUPS = HYBRID_THREADGROUP_SIZE / SIMD_WIDTH; // 8
constant uint HYBRID_V2_NUM_SIMD_GROUPS = HYBRID_V2_THREADGROUP_SIZE / SIMD_WIDTH; // 2

// Buffer header magic
// Atmosphere buffer offsets (must match buffers.rs atm_offsets exactly)
constant uint ATM_NUM_SHELLS            = 2;
constant uint ATM_NUM_WAVELENGTHS       = 3;
constant uint ATM_SHELLS_START          = 4;
constant uint ATM_SHELL_STRIDE          = 4;
constant uint ATM_OPTICS_START          = 260;   // 4 + 4*64
constant uint ATM_OPTICS_STRIDE         = 4;
constant uint ATM_ALBEDO_START          = 16708;  // 16644 + 64
constant uint ATM_REFRACTIVE_INDEX_START = 16772; // 16708 + 64 (v2)
constant uint ATM_CLOUD_EXT_START        = 16836; // 16772 + 64 (v3)
constant uint ATM_CLOUD_G_SCALED         = 16900; // 16836 + 64 (v3)

// Garstang constants
constant float H_RAYLEIGH = 8500.0f;
constant float H_AEROSOL  = 1500.0f;
constant float TAU_RAYLEIGH_550 = 0.0962f;

// Boundary nudge distance (meters). Must exceed f32 ULP at Earth radius
// (~0.5m) to guarantee shell crossing after radial nudge.
constant float BOUNDARY_NUDGE_M = 2.0f;

// SZA threshold (degrees) above which forced scattering activates.
// Below this, chains scatter naturally and rarely escape.
constant float ZENITH_SZA_START_DEG = 96.0f;

// Early-exit threshold for scout_tau_to_boundary.
// At tau > 20, 1-exp(-20) = 0.999999998 in f32. The forced-scattering
// weight is indistinguishable from 1.0 and we fall back to analog scatter.
constant float FORCED_TAU_CUTOFF = 20.0f;

// Maximum directional bias for the exponential transform.
// sigma' = sigma * (1 - alpha * cos_z). At alpha=0.5, sigma' in [0.5*sigma, 1.5*sigma].
constant float EXP_TRANSFORM_ALPHA_MAX = 0.5f;

// SZA at which the exponential transform reaches full strength.
constant float ZENITH_SZA_FULL_DEG = 106.0f;

// Power exponent for zenith-biased initial direction sampling.
// cos^n(theta_zenith) concentrates rays near zenith at deep twilight.
constant float ZENITH_BIAS_N = 5.0f;

// Maximum fraction of rays using zenith-biased sampling at deep twilight.
constant float ZENITH_MAX_FRACTION = 0.95f;

// Maximum fraction of zenith-allocated rays redirected to terminator lobe.
// At SZA >= 106: phase 5%, zenith 47.5%, terminator 47.5%.
constant float TERMINATOR_MAX_SHARE = 0.5f;

// Power-cosine exponent for the terminator lobe at maximum SZA.
constant float TERMINATOR_N_MAX = 8.0f;

// Tilt angle (degrees) of terminator axis from zenith at SZA_START / SZA_FULL.
constant float TERMINATOR_TILT_MIN_DEG = 20.0f;
constant float TERMINATOR_TILT_MAX_DEG = 60.0f;

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

inline float read_refractive_index(device const float* atm, uint shell_idx) {
    return atm[ATM_REFRACTIVE_INDEX_START + shell_idx];
}

inline float read_cloud_extinction(device const float* atm, uint shell_idx) {
    return atm[ATM_CLOUD_EXT_START + shell_idx];
}

// Eddington diffuse transmittance of accumulated (delta-scaled) cloud
// optical depth: T = 1/(1 + 0.75 tau (1 - g*)). Mirrors the CPU's
// AtmosphereModel::cloud_diffuse_transmittance — a diffusing deck
// transmits ~20-50%, which Beer-Lambert misrepresents by orders of
// magnitude (single-representation cloud transport).
inline float cloud_diffuse_transmittance(device const float* atm, float tau_cloud) {
    if (tau_cloud <= 0.0f) return 1.0f;
    float g = atm[ATM_CLOUD_G_SCALED];
    return 1.0f / (1.0f + 0.75f * tau_cloud * (1.0f - g));
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
// KBN (Kahan-Babuska-Neumaier) compensated summation
//
// Standard Kahan fails when the addend is larger than the running sum
// (the compensation captures the wrong rounding error). This happens at
// deep twilight when a single scatter event produces an energy spike
// exceeding the accumulated sum. Neumaier's variant compares magnitudes
// and always compensates the smaller operand, handling both cases.
// ============================================================================

struct KahanAccum {
    float sum;
    float comp; // compensation term

    KahanAccum() : sum(0.0f), comp(0.0f) {}
    KahanAccum(float s) : sum(s), comp(0.0f) {}

    void add(float value) {
        float t = sum + value;
        // Neumaier: compensate whichever operand is smaller in magnitude
        if (abs(sum) >= abs(value)) {
            comp += (sum - t) + value;
        } else {
            comp += (value - t) + sum;
        }
        sum = t;
    }

    float result() const { return sum + comp; }
};

// ============================================================================
// log1p polyfill -- MSL does not provide log1p(). For |x| > 0.5 we use
// log(1+x) directly (no cancellation risk). For |x| <= 0.5 we use the
// identity log1p(x) = x * log(1+x) / ((1+x) - 1) which recovers the
// lost low bits via the exact subtraction (1+x)-1 when |x| is small.
// ============================================================================

inline float metal_log1p(float x) {
    if (x > 0.5f || x < -0.5f) {
        return log(1.0f + x);
    }
    float u = 1.0f + x;
    float d = u - 1.0f;          // exact in f32 when |x| <= 0.5
    if (d == 0.0f) return x;     // x is subnormal or zero
    return log(u) * (x / d);
}

// ============================================================================
// xorshift64 RNG (Metal supports ulong natively on Apple Silicon)
// ============================================================================

inline ulong splitmix64(ulong state) {
    ulong z = state + 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

inline float xorshift_f32(thread ulong &state) {
    ulong x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    // Convert to f32 in (0, 1]: use top 24 bits for full mantissa precision.
    // Adding 1 before dividing guarantees the result is never exactly 0.0,
    // which prevents log(0) = -inf in free-path sampling.
    return float((x >> 40) + 1ul) * (1.0f / float((1ul << 24) + 1ul));
}

// ============================================================================
// Error-free transformations (DS arithmetic via FMA)
//
// two_product: exact split a*b = p + e using FMA.
// two_sum:     exact split a+b = s + e using Knuth's trick.
//
// These give ~14-digit discriminant precision in f32 by tracking the
// rounding error term explicitly. Applied in the ray-sphere intersection
// where the standard discriminant b^2 - a*c loses all significant digits
// for near-tangential rays at Earth scale.
// ============================================================================

struct DS {  // double-single: value = hi + lo
    float hi;
    float lo;
};

inline DS two_product(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);  // FMA keeps infinite intermediate precision
    return DS{p, e};
}

inline DS two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return DS{s, e};
}

inline DS ds_add(DS x, DS y) {
    DS s = two_sum(x.hi, y.hi);
    s.lo += x.lo + y.lo;
    // Renormalize
    DS r = two_sum(s.hi, s.lo);
    return r;
}

inline DS ds_sub(DS x, DS y) {
    DS neg_y = {-y.hi, -y.lo};
    return ds_add(x, neg_y);
}

// ============================================================================
// Ray-sphere intersection (DS discriminant + stable quadratic)
//
// The discriminant disc = b_half^2 - a*c is computed in double-single
// precision via FMA error-free transformations, giving ~14 significant
// digits instead of f32's ~7. This eliminates the precision collapse
// that causes spurious nonzero transmittance in deep-twilight shadow rays.
//
// Root finding uses the numerically stable formula:
//   q = -(b_half + copysign(sqrt_disc, b_half))
//   t1 = q / a,  t2 = c / q
// This avoids catastrophic cancellation when b_half and sqrt_disc have
// the same sign (one root comes from subtraction of nearly equal values).
// ============================================================================

struct RaySphereHit {
    float t_near;
    float t_far;
    bool hit;
};

inline RaySphereHit ray_sphere_intersect(float3 origin, float3 dir, float radius) {
    float a = dot(dir, dir);
    float b_half = dot(origin, dir);
    float r_pos = length(origin);
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

    float sqrt_disc = sqrt(max(disc, 0.0f));

    // Stable quadratic: avoid cancellation by choosing the sign of sqrt
    // that makes the sum largest in magnitude.
    float q = -(b_half + copysign(sqrt_disc, b_half));

    float t1, t2;
    if (abs(q) > 1e-30f) {
        t1 = q / a;
        t2 = c / q;
    } else {
        // Degenerate: ray origin on sphere surface, tangential
        float inv_a = 1.0f / a;
        t1 = (-b_half - sqrt_disc) * inv_a;
        t2 = (-b_half + sqrt_disc) * inv_a;
    }

    // Sort so t_near <= t_far
    result.t_near = min(t1, t2);
    result.t_far  = max(t1, t2);
    result.hit = true;
    return result;
}

// ============================================================================
// Shell index lookup -- O(log N) binary search
//
// Replaces the old O(N) linear scan. The binary search finds the largest
// shell index s such that r_inner[s] <= radius. Since shells are contiguous
// (r_outer[s] == r_inner[s+1]), this guarantees the radius is in shell s.
// ============================================================================

inline int shell_index_binary(device const float* atm, float r) {
    uint ns = atm_num_shells(atm);
    if (ns == 0) return -1;

    // Bounds check
    float r_inner_first = atm[ATM_SHELLS_START];
    float r_outer_last = atm[ATM_SHELLS_START + (ns - 1) * ATM_SHELL_STRIDE + 1];
    if (r < r_inner_first || r >= r_outer_last) return -1;

    // Binary search: find largest s with r_inner[s] <= r
    uint lo = 0;
    uint hi = ns;
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        float r_inner_mid = atm[ATM_SHELLS_START + mid * ATM_SHELL_STRIDE];
        if (r_inner_mid <= r) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo == 0) ? -1 : int(lo - 1);
}

// Surface and top-of-atmosphere radii from the PACKED SHELLS — the single
// source of truth, exactly matching the CPU's AtmosphereModel accessors
// (surface_radius = r_inner of shell 0, toa_radius = r_outer of the last
// shell). The old hardcoded EARTH_RADIUS_M + TOA_ALTITUDE_M (100 km)
// silently truncated the 150 km USSA-76 thermosphere extension — the very
// layer that carries the deep-twilight (SZA >= 104) signal.
inline float atm_surface_radius(device const float* atm) {
    uint ns = atm_num_shells(atm);
    return (ns == 0) ? (EARTH_RADIUS_M)
                     : atm[ATM_SHELLS_START];
}

inline float atm_toa_radius(device const float* atm) {
    uint ns = atm_num_shells(atm);
    return (ns == 0) ? (EARTH_RADIUS_M + TOA_ALTITUDE_M)
                     : atm[ATM_SHELLS_START + (ns - 1) * ATM_SHELL_STRIDE + 1];
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
    // rsqrt + multiply is faster than sqrt + divide on Metal ALU
    float inv_sqrt_d = rsqrt(max(denom, 1e-20f));
    return (1.0f - g2) * inv_sqrt_d * inv_sqrt_d * inv_sqrt_d;
}

inline float mixed_phase(float cos_theta, ShellOptics op) {
    // Exact mixture, ALWAYS: a former rf > 0.99 pure-Rayleigh shortcut
    // mismatched the seed sampler (which draws HG with prob 1-rf for any
    // rf < 1), biasing the mixture-MIS seed weights (mirrors the CPU fix).
    return op.rayleigh_fraction * rayleigh_phase(cos_theta)
         + (1.0f - op.rayleigh_fraction) * henyey_greenstein_phase(cos_theta, op.asymmetry);
}

// ============================================================================
// Stokes [I,Q,U,V] polarized RT helpers
//
// For our atmosphere (Rayleigh + spherical HG aerosols):
//   P22 = P11, P44 = P33 (Rayleigh symmetry)
//   P34 = 0 (spherical particles, diagonal HG)
//
// The combined Mueller matrix M_scatter * R(phi) reduces to 4 scalar
// equations with 3 parameters (A, B, C), costing 14 FP ops per scatter
// vs 64 for a full 4x4 matmul. Zero accuracy loss.
// ============================================================================

// Rayleigh P12 element: polarization coupling (off-diagonal)
inline float rayleigh_P12(float cos_theta) {
    float sin2 = 1.0f - cos_theta * cos_theta;
    return -0.75f * sin2;  // = -(3/4)*sin^2(theta)
}

// Rayleigh P33 element: circular polarization coupling
inline float rayleigh_P33(float cos_theta) {
    return 1.5f * cos_theta;  // = (3/2)*cos(theta)
}

// Trig-free scattering plane rotation angle
//
// Computes cos(2*phi) and sin(2*phi) for the rotation between successive
// scattering planes WITHOUT any trig calls. Uses cross/dot products of
// direction vectors + double-angle identities.
//
// dir_in:   incoming direction (before current scatter)
// dir_out:  outgoing direction (after current scatter = current propagation)
// dir_next: direction after next scatter (or sun direction for NEE)
//
// When dir_in and dir_out are (anti-)parallel (forward/back scatter),
// the scattering plane is undefined. Returns cos2phi=1, sin2phi=0 (no rotation).
inline void scattering_plane_rotation(float3 dir_in, float3 dir_out, float3 dir_next,
                                       thread float &cos2phi, thread float &sin2phi) {
    float3 n1 = cross(dir_in, dir_out);   // normal to old scattering plane
    float3 n2 = cross(dir_out, dir_next); // normal to new scattering plane

    float n1_sq = dot(n1, n1);
    float n2_sq = dot(n2, n2);

    if (n1_sq < 1e-20f || n2_sq < 1e-20f) {
        // Degenerate: forward/backward scatter, no rotation needed
        cos2phi = 1.0f;
        sin2phi = 0.0f;
        return;
    }

    float inv_norm = rsqrt(n1_sq * n2_sq);
    float cos_phi = dot(n1, n2) * inv_norm;
    float sin_phi = dot(dir_out, cross(n1, n2)) * inv_norm;

    // Clamp cos_phi to avoid NaN from numerical noise
    cos_phi = clamp(cos_phi, -1.0f, 1.0f);

    // Double-angle identities: cos(2phi) = 2cos^2(phi) - 1
    //                          sin(2phi) = 2sin(phi)cos(phi)
    cos2phi = 2.0f * cos_phi * cos_phi - 1.0f;
    sin2phi = 2.0f * sin_phi * cos_phi;
}

// Compute the 3 Mueller parameters for mixed Rayleigh+HG scattering
//   A = alpha*P11_R + (1-alpha)*P11_HG   (= scalar mixed phase function)
//   B = alpha*P12_R                       (polarization coupling)
//   C = alpha*P33_R + (1-alpha)*P11_HG    (circular polarization)
inline void stokes_ABC(float cos_theta, ShellOptics op,
                       thread float &A, thread float &B, thread float &C) {
    float alpha = op.rayleigh_fraction;
    float p11_r = rayleigh_phase(cos_theta);
    float p12_r = rayleigh_P12(cos_theta);
    float p33_r = rayleigh_P33(cos_theta);
    float p11_hg = henyey_greenstein_phase(cos_theta, op.asymmetry);

    A = alpha * p11_r + (1.0f - alpha) * p11_hg;
    B = alpha * p12_r;
    C = alpha * p33_r + (1.0f - alpha) * p11_hg;
}

// Apply the analytically unrolled Mueller matrix to a Stokes vector.
// 4 equations, 14 FP ops:
//   I' = A*I + B*(c2*Q + s2*U)
//   Q' = B*I + A*(c2*Q + s2*U)    [D=A for Rayleigh symmetry]
//   U' = C*(c2*U - s2*Q)
//   V' = C*V                       [E=C for Rayleigh symmetry]
inline float4 scatter_stokes(float A, float B, float C,
                              float cos2phi, float sin2phi, float4 s_in) {
    float rotQU = cos2phi * s_in.y + sin2phi * s_in.z;  // c2*Q + s2*U
    float4 s_out;
    s_out.x = A * s_in.x + B * rotQU;        // I'
    s_out.y = B * s_in.x + A * rotQU;        // Q'  (D=A)
    s_out.z = C * (cos2phi * s_in.z - sin2phi * s_in.y);  // U'
    s_out.w = C * s_in.w;                     // V'  (E=C)
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

ShellBoundary next_shell_boundary(float3 pos, float3 dir, float r_inner, float r_outer) {
    ShellBoundary result;
    result.found = false;
    result.dist = 1e30f;
    result.is_outward = true;

    const float EPS = 1e-5f;

    RaySphereHit outer = ray_sphere_intersect(pos, dir, r_outer);
    if (outer.hit) {
        // Compute inner sphere test once for both branches below.
        RaySphereHit inner = ray_sphere_intersect(pos, dir, r_inner);

        if (outer.t_near > EPS) {
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

        // outer hit but neither t_near nor t_far usable -- check inner
        if (inner.hit && inner.t_near > EPS) {
            result.dist = inner.t_near;
            result.is_outward = false;
            result.found = true;
        }
        return result;
    }

    // No outer hit: inner sphere only
    RaySphereHit inner = ray_sphere_intersect(pos, dir, r_inner);
    if (inner.hit && inner.t_near > EPS) {
        result.dist = inner.t_near;
        result.is_outward = false;
        result.found = true;
    }
    return result;
}

// ============================================================================
// Snell's law refraction at a spherical shell boundary
//
// Returns the new ray direction after refraction. For total internal
// reflection (TIR), returns the reflected direction. When n_from == n_to,
// returns the original direction (fast path).
// ============================================================================

// Refract with known radius -- avoids rsqrt inside normalize() since the
// position was already snapped to target_r.
float3 refract_at_boundary_r(float3 dir, float3 boundary_pos, float inv_r, float n_from, float n_to) {
    if (abs(n_from - n_to) < 1e-7f) return dir;

    float3 outward = boundary_pos * inv_r;

    float cos_dir_normal = dot(dir, outward);
    float3 normal = (cos_dir_normal < 0.0f) ? outward : -outward;

    float cos_i = -dot(dir, normal);
    float eta = n_from / n_to;
    float k = fma(-eta * eta, fma(-cos_i, cos_i, 1.0f), 1.0f);

    if (k < 0.0f) {
        return fma(normal, 2.0f * cos_i, dir);
    }

    float cos_t = sqrt(k);
    return fma(normal, fma(eta, cos_i, -cos_t), dir * eta);
}

float3 refract_at_boundary(float3 dir, float3 boundary_pos, float n_from, float n_to) {
    if (abs(n_from - n_to) < 1e-7f) return dir;

    float3 outward = normalize(boundary_pos);

    // Orient normal to face the incoming ray
    float cos_dir_normal = dot(dir, outward);
    float3 normal = (cos_dir_normal < 0.0f) ? outward : -outward;

    float cos_i = -dot(dir, normal);
    float eta = n_from / n_to;
    float k = 1.0f - eta * eta * (1.0f - cos_i * cos_i);

    if (k < 0.0f) {
        // Total internal reflection: result is unit by reflection identity.
        return dir + normal * (2.0f * cos_i);
    }

    float cos_t = sqrt(k);
    float factor = eta * cos_i - cos_t;
    // Snell refraction: result is unit by Snell's law identity.
    return dir * eta + normal * factor;
}

// ============================================================================
// Radial boundary nudge
//
// At Earth scale (r ~ 6.4e6 m), f32 ULP is ~0.5m. For tangential rays at
// shell boundaries, nudging along the ray direction by 1m produces ZERO
// radial movement in f32 (the tangential component dominates, and the radial
// delta rounds to zero). The position stays at exactly the boundary radius,
// causing the ray to get stuck in an infinite loop.
//
// Fix: nudge RADIALLY (along the outward normal from Earth center) by 2m.
// This guarantees the position crosses into the next shell regardless of
// ray direction.
// ============================================================================

// Snap position to exact target radius. Prevents cumulative f32 position
// drift from pos + dir * dist not landing exactly on the shell boundary.
// Without this, over ~50 boundary crossings the radius error grows to
// ~50 * ULP(6.4e6) = ~25m, placing the photon in the wrong shell.
inline float3 snap_to_radius(float3 pos, float target_r) {
    float r = length(pos);
    return (r > 0.0f) ? pos * (target_r / r) : pos;
}

inline float3 radial_nudge(float3 boundary_pos, bool is_outward) {
    float bp_r = length(boundary_pos);
    float3 radial_dir = (bp_r > 1e-10f) ? boundary_pos / bp_r : float3(1.0f, 0.0f, 0.0f);
    float nudge_sign = is_outward ? 1.0f : -1.0f;
    return boundary_pos + radial_dir * (nudge_sign * BOUNDARY_NUDGE_M);
}

// ============================================================================
// Shadow ray transmittance -- shell-by-shell with refraction
//
// Traces a ray from scatter_pos toward the sun through the atmosphere,
// accumulating optical depth with Kahan summation. At each shell boundary,
// applies Snell's law refraction and radial nudge.
//
// This replaces the old analytical ray_path_through_shell approach, which
// assumed straight rays (no refraction). The CPU ground truth is in
// twilight-core/src/single_scatter.rs:shadow_ray_transmittance.
// ============================================================================

float shadow_ray_transmittance(device const float* atm, float3 start_pos,
                                float3 sun_dir, uint wl_idx) {
    uint ns = atm_num_shells(atm);
    float surface_radius = atm[ATM_SHELLS_START]; // r_inner of shell 0

    // ── Umbra cylinder culling (O(1) pre-check) ────────────────────────
    float p_proj = dot(start_pos, sun_dir);
    if (p_proj < 0.0f) {
        float3 cross_ps = cross(start_pos, sun_dir);
        float perp_dist_sq = dot(cross_ps, cross_ps);
        if (perp_dist_sq < surface_radius * surface_radius) {
            return 0.0f;
        }
    }

    float3 pos = start_pos;
    float3 dir = sun_dir;

    // Plain float accumulation for shadow tau -- Kahan is overkill here
    // because we only need ~3 significant digits (exp(-tau) dynamic range)
    // and the early-exit at tau > 50 means we never accumulate more than
    // ~50 terms of similar magnitude. Saves 3 FP ops per shell crossing.
    float tau = 0.0f;
    float tau_cloud = 0.0f;

    int sidx = shell_index_binary(atm, length(pos));
    if (sidx < 0) return 1.0f;
    uint us = uint(sidx);

    for (uint iter = 0; iter < 200; iter++) {
        uint shell_base = ATM_SHELLS_START + us * ATM_SHELL_STRIDE;
        float r_inner = atm[shell_base];
        float r_outer = atm[shell_base + 1];

        // Inline extinction read (avoid full ShellOptics load)
        float extinction = atm[ATM_OPTICS_START + (us * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) break;

        tau += extinction * bnd.dist;
        tau_cloud += read_cloud_extinction(atm, us) * bnd.dist;

        // Snap + nudge -- avoid redundant length() by reusing snap_to_radius
        float3 boundary_pos = pos + dir * bnd.dist;
        float target_r = bnd.is_outward ? r_outer : r_inner;
        float bp_r = length(boundary_pos);
        if (bp_r > 0.0f) boundary_pos *= (target_r / bp_r);

        // Refract (use _r variant: position already snapped to target_r)
        float inv_target_r = 1.0f / target_r;
        float n_from = read_refractive_index(atm, us);
        uint next_shell = bnd.is_outward ? us + 1 : us - 1;
        float n_to = (next_shell < ns) ? read_refractive_index(atm, next_shell) : 1.0f;
        dir = refract_at_boundary_r(dir, boundary_pos, inv_target_r, n_from, n_to);

        // Radial nudge (reuse inv_target_r)
        float3 radial = boundary_pos * inv_target_r;
        float nudge_sign = bnd.is_outward ? 1.0f : -1.0f;
        pos = boundary_pos + radial * (nudge_sign * BOUNDARY_NUDGE_M);

        // Ground hit (use target_r which is exact after snap)
        if (!bnd.is_outward && target_r <= surface_radius + 1.0f) {
            return 0.0f;
        }

        if (next_shell >= ns) break;
        us = next_shell;

        if (tau > 50.0f) return 0.0f;
    }

    // fast::exp is sufficient for shadow transmittance -- we only need
    // ~3 significant digits (the MC weight noise dominates).
    // Clear-air Beer-Lambert x Eddington diffuse for the cloud portion.
    return fast::exp(-tau) * cloud_diffuse_transmittance(atm, tau_cloud);
}

// ============================================================================
// Sampling functions
// ============================================================================

// Newton-Raphson cube root: ~6 cycles vs ~16 for pow(x, 1/3).
// 2 iterations of x = (2x + a/(x*x))/3 from an rsqrt seed.
inline float fast_cbrt(float a) {
    float x = abs(a);
    if (x < 1e-30f) return 0.0f;
    // Seed: cbrt(x) ≈ x^(1/3) ≈ x * rsqrt(x)^(2/3). Use rsqrt as
    // rough seed then refine. Initial guess via bit hack:
    float y = as_type<float>((as_type<uint>(x) / 3u) + 0x2a508bdb);
    // Halley iteration (cubic convergence, 2 iterations for f32 precision)
    y = y * (2.0f / 3.0f) + x / (3.0f * y * y);
    y = y * (2.0f / 3.0f) + x / (3.0f * y * y);
    return copysign(y, a);
}

inline float sample_rayleigh_analytic(float xi) {
    float q = 8.0f * xi - 4.0f;
    float disc = fma(q * q, 0.25f, 1.0f);
    float sqrt_disc = sqrt(disc);
    float a_val = -q * 0.5f + sqrt_disc;
    float b_val = -q * 0.5f - sqrt_disc;
    float mu = fast_cbrt(a_val) + fast_cbrt(b_val);
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
    float cos_phi;
    float sin_phi = sincos(phi, cos_phi);

    float3 w = dir;
    float3 up = (abs(w.z) < 0.9f) ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 u_vec = normalize(cross(w, up));
    float3 v_vec = cross(w, u_vec);

    // (u_vec, v_vec, w) is orthonormal: result is unit length, no normalize needed.
    float sc = sin_theta * cos_phi;
    float ss = sin_theta * sin_phi;
    return sc * u_vec + ss * v_vec + cos_theta * w;
}

float3 sample_hemisphere(float3 normal, thread ulong &rng) {
    float xi1 = xorshift_f32(rng);
    float xi2 = xorshift_f32(rng);
    float cos_theta = sqrt(xi1);
    float phi = 2.0f * PI * xi2;
    return scatter_direction(normal, cos_theta, phi);
}

// Sample from power-cosine distribution biased toward normal.
// PDF: p(omega) = (n+1)/(2*pi) * cos^n(theta).
// Consumes exactly 2 RNG draws (matching sample_hemisphere).
// Returns (direction, cos_theta).
struct ZenithSample {
    float3 dir;
    float cos_theta;
};

ZenithSample sample_zenith_biased(float3 normal, float n, thread ulong &rng) {
    float xi1 = xorshift_f32(rng);
    float xi2 = xorshift_f32(rng);
    float cos_theta = pow(xi1, 1.0f / (n + 1.0f));
    float phi = 2.0f * PI * xi2;
    float3 dir = scatter_direction(normal, cos_theta, phi);
    return ZenithSample{dir, cos_theta};
}

// PDF over solid angle of the (untruncated) power-cosine lobe drawn by
// sample_zenith_biased: p(omega) = (n+1) cos^n(theta) / (2 pi) on the
// upper hemisphere about the axis, 0 below. (The GPU sampler is
// untruncated; this pdf matches it exactly. Unbiasedness needs only
// sampler == pdf; weight boundedness comes from the mixture's phase
// component, NOT from truncation — so no weight clamps and no NaN-prone
// unbounded importance ratios, the root cause of the old v2 fireflies.)
inline float power_cos_pdf(float cos_theta, float n) {
    if (cos_theta <= 0.0f) return 0.0f;
    return (n + 1.0f) * pow(cos_theta, n) / (2.0f * PI);
}

// Density of the 3-component seed mixture at omega (mirrors the CPU's
// seed_mixture_pdf): q = a_p*P(omega.sun)/4pi + a_z*pcos(up) + a_t*pcos(term).
inline float seed_mixture_pdf(float3 omega, float3 sun_dir, float3 local_up,
                              float3 term_axis, float alpha_p, float alpha_z,
                              float alpha_t, float n_zenith, float m_term,
                              ShellOptics op) {
    float q = alpha_p * mixed_phase(dot(omega, sun_dir), op) * INV_4PI;
    if (alpha_z > 1e-6f) q += alpha_z * power_cos_pdf(dot(omega, local_up), n_zenith);
    if (alpha_t > 1e-6f) q += alpha_t * power_cos_pdf(dot(omega, term_axis), m_term);
    return q;
}

// 3-branch direction sampling parameters, SZA-adaptive.
struct BranchParams {
    float zenith_frac;  // total non-phase fraction
    float n_zenith;     // power-cosine exponent for zenith lobe
    float term_share;   // fraction of zenith-allocated rays -> terminator
    float m_term;       // power-cosine exponent for terminator lobe
    float tilt_rad;     // tilt angle of terminator axis from zenith
};

struct SecondarySetup {
    float3 local_up;
    float3 term_axis_dir;
    float alpha_p;
    float alpha_z;
    float alpha_t;
    float n_zenith;
    float m_term;
    float alpha_et;
    uint use_forced;
};

inline BranchParams branch_params_for_sza(float sza_deg) {
    float sza_t = clamp((sza_deg - ZENITH_SZA_START_DEG)
                        / (ZENITH_SZA_FULL_DEG - ZENITH_SZA_START_DEG), 0.0f, 1.0f);
    BranchParams bp;
    bp.zenith_frac = 0.5f + (ZENITH_MAX_FRACTION - 0.5f) * sza_t;
    bp.n_zenith = 1.0f + (ZENITH_BIAS_N - 1.0f) * sza_t;
    bp.term_share = TERMINATOR_MAX_SHARE * sza_t;
    bp.m_term = 1.0f + (TERMINATOR_N_MAX - 1.0f) * sza_t;
    float tilt_deg = TERMINATOR_TILT_MIN_DEG
        + (TERMINATOR_TILT_MAX_DEG - TERMINATOR_TILT_MIN_DEG) * sza_t;
    bp.tilt_rad = tilt_deg * PI / 180.0f;
    return bp;
}

// Compute terminator axis: unit vector tilted from up toward sub-solar horizon.
inline float3 terminator_axis(float3 up, float3 sun_dir, float tilt_rad) {
    float dot_us = dot(sun_dir, up);
    float3 horiz = sun_dir - dot_us * up;
    float h_len = length(horiz);
    if (h_len < 1e-6f) {
        return up;
    }
    float3 sun_horiz = horiz / h_len;
    float cos_t;
    float sin_t = sincos(tilt_rad, cos_t);
    return normalize(cos_t * up + sin_t * sun_horiz);
}


// ============================================================================
// Kernel 1: single_scatter_spectrum
//
// One thread per wavelength. Full LOS integration with refracted shadow rays.
// Kahan summation for optical depth and radiance accumulation.
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

    float toa_radius = atm_toa_radius(atm);
    float surface_radius = atm_surface_radius(atm);

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

    KahanAccum radiance;
    KahanAccum tau_obs;
    // Cloud portion of the eye path: Eddington diffuse (broadband).
    float tau_cloud_obs = 0.0f;

    float cos_theta = dot(sun_dir, view_dir);

    for (uint step = 0; step < num_steps; step++) {
        float s = (float(step) + 0.5f) * ds;
        float3 scatter_pos = observer_pos + view_dir * s;
        float r = length(scatter_pos);

        if (r > toa_radius || r < surface_radius) continue;

        int sidx = shell_index_binary(atm, r);
        if (sidx < 0) continue;

        ShellOptics op = read_optics(atm, uint(sidx), wl_idx);
        float cloud_ext_step = read_cloud_extinction(atm, uint(sidx));
        float beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30f) {
            tau_obs.add(op.extinction * ds);
            tau_cloud_obs += cloud_ext_step * ds;
            continue;
        }

        // Single exp(-(tau + half_step)) is both faster and more precise
        // than the product of two separate exps (which rounds the f32
        // multiply and wastes an ALU slot).
        float tau_cloud_mid = tau_cloud_obs + cloud_ext_step * ds * 0.5f;
        float t_obs = exp(-(tau_obs.result() + op.extinction * ds * 0.5f))
                    * cloud_diffuse_transmittance(atm, tau_cloud_mid);

        if (t_obs < 1e-30f) break;

        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);

        if (t_sun < 1e-30f) {
            tau_obs.add(op.extinction * ds);
            tau_cloud_obs += cloud_ext_step * ds;
            continue;
        }

        float phase = mixed_phase(cos_theta, op);
        float di = beta_scat * phase / (4.0f * PI) * t_sun * t_obs * ds;
        radiance.add(di);

        tau_obs.add(op.extinction * ds);
        tau_cloud_obs += cloud_ext_step * ds;
    }

    // Ground reflection (Lambertian BRDF = albedo / pi)
    if (hits_ground) {
        float albedo = read_albedo(atm, wl_idx);
        if (albedo > 1e-10f) {
            float3 ground_pos = observer_pos + view_dir * los_end;
            float3 ground_normal = normalize(ground_pos);
            float cos_sun_incidence = dot(sun_dir, ground_normal);

            if (cos_sun_incidence > 0.0f) {
                float t_sun_ground = shadow_ray_transmittance(atm, ground_pos, sun_dir, wl_idx);
                float t_obs_ground = exp(-tau_obs.result())
                    * cloud_diffuse_transmittance(atm, tau_cloud_obs);
                radiance.add(albedo / PI * cos_sun_incidence * t_sun_ground * t_obs_ground);
            }
        }
    }

    output[tid] = radiance.result();
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

    float surface_radius = atm_surface_radius(atm);

    float3 pos = observer_pos;
    float3 dir = view_dir;
    float3 prev_dir = dir; // for Stokes scattering plane tracking
    float4 stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
    float weight = 1.0f;
    KahanAccum result_weight;

    for (uint bounce = 0; bounce < MAX_SCATTERS; bounce++) {
        float r = length(pos);
        int sidx = shell_index_binary(atm, r);
        if (sidx < 0) break;

        uint us = uint(sidx);
        ShellGeom sh = read_shell(atm, us);
        ShellOptics op = read_optics(atm, us, wl_idx);

        if (op.extinction < 1e-20f) {
            ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) break;
            float3 boundary_pos = pos + dir * bnd.dist;
            boundary_pos = snap_to_radius(boundary_pos, bnd.is_outward ? sh.r_outer : sh.r_inner);
            float n_from = read_refractive_index(atm, us);
            uint next_s = bnd.is_outward ? us + 1 : us - 1;
            float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
            dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        // Sample free path
        float xi = xorshift_f32(rng);
        // -log(xi) with xi in (0,1] is equivalent to -log(1-U) with U in [0,1)
        // since 1-U has the same distribution as U. Avoids the log1p polyfill
        // branch and extra ops. xi is guaranteed > 0 by the RNG.
        float free_path = -log(xi) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            // Exit shell without scattering
            float3 boundary_pos = pos + dir * bnd.dist;
            boundary_pos = snap_to_radius(boundary_pos, bnd.is_outward ? sh.r_outer : sh.r_inner);

            // Ground reflection: depolarizes
            if (!bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                float3 normal = normalize(boundary_pos);
                prev_dir = dir;
                dir = sample_hemisphere(normal, rng);
                pos = radial_nudge(boundary_pos, true);
                stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
                continue;
            }

            // Refract and nudge past boundary
            {
                float n_from = read_refractive_index(atm, us);
                uint next_s = bnd.is_outward ? us + 1 : us - 1;
                float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
                dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
            }
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        // Scattering event
        pos = pos + dir * free_path;

        // NEE: apply Mueller to photon's current Stokes state
        float t_sun = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun > 1e-30f) {
            float cos_angle = dot(sun_dir, dir);
            float A_nee, B_nee, C_nee;
            stokes_ABC(cos_angle, op, A_nee, B_nee, C_nee);
            float cos2phi_nee, sin2phi_nee;
            scattering_plane_rotation(prev_dir, dir, -sun_dir, cos2phi_nee, sin2phi_nee);
            float4 nee_stokes = scatter_stokes(A_nee, B_nee, C_nee, cos2phi_nee, sin2phi_nee, stokes);
            result_weight.add(weight * t_sun * nee_stokes.x / (4.0f * PI));
        }

        // Apply SSA
        weight *= op.ssa;

        // Sample new direction and update Stokes state
        float cos_theta;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        float3 new_dir = scatter_direction(dir, cos_theta, phi);

        // Update Stokes through this scatter
        float A_s, B_s, C_s;
        stokes_ABC(cos_theta, op, A_s, B_s, C_s);
        float cos2phi_s, sin2phi_s;
        scattering_plane_rotation(prev_dir, dir, new_dir, cos2phi_s, sin2phi_s);
        stokes = scatter_stokes(A_s, B_s, C_s, cos2phi_s, sin2phi_s, stokes);
        if (stokes.x > 1e-30f) {
            stokes *= 1.0f / stokes.x;
        } else {
            stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
        }

        prev_dir = dir;
        dir = new_dir;
    }

    output[tid] = result_weight.result();
}

// ============================================================================
// Forced scattering scout: compute tau_max to atmosphere boundary
//
// Marches shell-by-shell with refraction. Early-exits at tau > FORCED_TAU_CUTOFF
// (20.0) since the weight correction is indistinguishable from 1.0 at that point.
// Returns (tau_max, hit_ground). When hit_ground is true, forced scattering
// should NOT be used -- the analog loop handles ground reflection.
// ============================================================================

struct ScoutResult {
    float tau;
    bool  hit_ground;
};

ScoutResult scout_tau_to_boundary(device const float* atm, float3 start_pos,
                                   float3 start_dir, uint wl_idx) {
    uint ns = atm_num_shells(atm);
    float surface_radius = atm[ATM_SHELLS_START];
    float3 pos = start_pos;
    float3 dir = start_dir;
    float tau = 0.0f;

    int sidx = shell_index_binary(atm, length(pos));
    if (sidx < 0) return ScoutResult{0.0f, false};
    uint us = uint(sidx);

    for (uint iter = 0; iter < 200; iter++) {
        uint shell_base = ATM_SHELLS_START + us * ATM_SHELL_STRIDE;
        float r_inner = atm[shell_base];
        float r_outer = atm[shell_base + 1];
        float extinction = atm[ATM_OPTICS_START + (us * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) return ScoutResult{tau, false};

        tau += extinction * bnd.dist;

        float3 boundary_pos = pos + dir * bnd.dist;
        float target_r = bnd.is_outward ? r_outer : r_inner;
        float bp_r = length(boundary_pos);
        if (bp_r > 0.0f) boundary_pos *= (target_r / bp_r);

        float inv_target_r = 1.0f / target_r;
        float n_from = read_refractive_index(atm, us);
        uint next_shell = bnd.is_outward ? us + 1 : us - 1;
        float n_to = (next_shell < ns) ? read_refractive_index(atm, next_shell) : 1.0f;
        dir = refract_at_boundary_r(dir, boundary_pos, inv_target_r, n_from, n_to);

        float3 radial = boundary_pos * inv_target_r;
        float nudge_sign = bnd.is_outward ? 1.0f : -1.0f;
        pos = boundary_pos + radial * (nudge_sign * BOUNDARY_NUDGE_M);

        if (!bnd.is_outward && target_r <= surface_radius + 1.0f) {
            return ScoutResult{tau, true};
        }
        if (next_shell >= ns) return ScoutResult{tau, false};
        us = next_shell;

        if (tau > FORCED_TAU_CUTOFF) return ScoutResult{tau, false};
    }

    return ScoutResult{tau, false};
}

// ============================================================================
// Forced scattering helper: advance along a ray to a target optical depth
//
// Marches shell-by-shell with refraction, consuming tau_target of optical
// depth. Returns the scatter position, arrival direction, and shell index.
// Called only on escape events (rare), so the cost is amortized.
// ============================================================================

struct AdvanceResult {
    float3 pos;
    float3 dir;
    uint   shell_idx;
};

AdvanceResult advance_to_optical_depth(device const float* atm, float3 start_pos,
                                        float3 start_dir, float tau_target,
                                        uint wl_idx) {
    uint ns = atm_num_shells(atm);
    float surface_radius = atm[ATM_SHELLS_START]; // r_inner of shell 0
    float3 pos = start_pos;
    float3 dir = start_dir;
    float tau_acc = 0.0f;

    int sidx = shell_index_binary(atm, length(pos));
    if (sidx < 0) return AdvanceResult{pos, dir, 0};
    uint us = uint(sidx);

    for (uint iter = 0; iter < 200; iter++) {
        float r_inner = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE];
        float r_outer = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE + 1];
        uint optics_idx = us * MAX_WAVELENGTHS + wl_idx;
        float extinction = atm[ATM_OPTICS_START + optics_idx * ATM_OPTICS_STRIDE];

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) return AdvanceResult{pos, dir, us};

        float tau_shell = extinction * bnd.dist;

        if (tau_acc + tau_shell >= tau_target) {
            // Scatter point is within this shell
            float tau_remaining = tau_target - tau_acc;
            float dist = (extinction > 1e-30f) ? tau_remaining / extinction : bnd.dist;
            pos = pos + dir * dist;
            return AdvanceResult{pos, dir, us};
        }

        // Cross boundary
        tau_acc += tau_shell;
        float3 boundary_pos = pos + dir * bnd.dist;
        boundary_pos = snap_to_radius(boundary_pos, bnd.is_outward ? r_outer : r_inner);
        float n_from = read_refractive_index(atm, us);
        uint next_shell = bnd.is_outward ? us + 1 : us - 1;
        float n_to = (next_shell < ns) ? read_refractive_index(atm, next_shell) : 1.0f;
        dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
        pos = radial_nudge(boundary_pos, bnd.is_outward);

        // Hit ground
        if (!bnd.is_outward && length(pos) <= surface_radius + 1.0f) {
            return AdvanceResult{pos, dir, us};
        }
        // Exited atmosphere
        if (next_shell >= ns) return AdvanceResult{pos, dir, us};
        us = next_shell;
    }

    return AdvanceResult{pos, dir, us};
}

// ============================================================================
// Secondary chain tracer (used by hybrid_scatter kernel)
//
// Full Stokes [I,Q,U,V] propagation through the secondary chain.
// Tracks the photon's polarization state (normalized, I=1) through
// each scatter event. At each NEE, applies the Mueller matrix to the
// photon's actual Stokes state, not to unpolarized [1,0,0,0].
//
// This captures the B*(Q/I)*cos(2phi) polarization-intensity coupling
// that can be up to ~5-10% of the multi-scatter contribution at specific
// twilight geometries.
//
// Returns float4: [I, Q, U, V] total Stokes contribution.
// ============================================================================

float4 trace_secondary_chain(device const float* atm, float3 start_pos,
                             float3 sun_dir, uint wl_idx,
                             ShellOptics start_optics, float3 prev_dir_in,
                             SecondarySetup setup,
                             uint ray_idx, uint total_rays,
                             thread ulong &rng) {
    float surface_radius = atm_surface_radius(atm);

    // Unbiased one-sample-MIS seed (port of the CPU estimator): sample
    // omega from the 3-component mixture, weight by
    //   w0 = P(omega.view)/4pi / q(omega)
    // identically for every branch — the balance-heuristic estimator.
    // Samplers and RNG consumption order unchanged. This replaces the old
    // per-branch heuristic weights whose unbounded importance ratios were
    // the v2 firefly/NaN source.
    float xi_jitter = xorshift_f32(rng);
    float xi_mix = (float(ray_idx) + xi_jitter) / max(float(total_rays), 1.0f);
    float3 dir;
    if (xi_mix < setup.alpha_p) {
        float ct;
        if (xorshift_f32(rng) < start_optics.rayleigh_fraction) {
            ct = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            ct = sample_henyey_greenstein(xorshift_f32(rng), start_optics.asymmetry);
        }
        float phi_init = 2.0f * PI * xorshift_f32(rng);
        dir = scatter_direction(sun_dir, ct, phi_init);
    } else if (xi_mix < setup.alpha_p + setup.alpha_z || setup.alpha_t < 1e-6f) {
        ZenithSample zs = sample_zenith_biased(setup.local_up, setup.n_zenith, rng);
        dir = zs.dir;
    } else {
        ZenithSample zs = sample_zenith_biased(setup.term_axis_dir, setup.m_term, rng);
        dir = zs.dir;
    }

    float q_seed = seed_mixture_pdf(dir, sun_dir, setup.local_up,
                                    setup.term_axis_dir, setup.alpha_p,
                                    setup.alpha_z, setup.alpha_t,
                                    setup.n_zenith, setup.m_term, start_optics);
    // prev_dir_in is the LOS view direction: the physical seed-scatter
    // cosine is omega.view (matches the CPU convention).
    float w0 = (q_seed > 1e-30f)
        ? mixed_phase(dot(dir, prev_dir_in), start_optics) * INV_4PI / q_seed
        : 0.0f;

    // Seed polarization: unpolarized (the exact treatment would Mueller-
    // rotate the omega->view seed scatter; multiply-scattered light is
    // weakly polarized and the I-error is sub-percent — same approximation
    // as the CPU chain).
    float4 stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);

    float3 pos = start_pos;
    float3 current_dir = dir;
    float3 prev_dir = sun_dir; // direction before current propagation segment
    // NOTE: no start_optics.ssa factor — the host-side integrator's
    // beta_scat at the seed point already carries it (double-count removed,
    // mirroring the CPU fix).
    float weight = w0;

    KahanAccum total_I, total_Q, total_U, total_V;

    for (uint scatter_iter = 0; scatter_iter < HYBRID_MAX_BOUNCES; scatter_iter++) {
        // --- Decide scatter mode for this bounce ---
        bool forced_this_bounce = false;
        float tau_max = 0.0f;

        if (setup.use_forced != 0u) {
            ScoutResult scout = scout_tau_to_boundary(atm, pos, current_dir, wl_idx);
            tau_max = scout.tau;
            // Force scatter only when path exits to space, optical depth is
            // within the useful range, and tau >= forced_tau_min. Without the
            // lower bound, chains at high altitude (tau ~ 1e-5) get killed by
            // weight *= (1 - exp(-tau)) ~ tau, losing 5 orders of magnitude
            // per bounce. The CPU falls back to analog mode for small tau.
            float forced_tau_min = (setup.alpha_et > 0.3f) ? 0.02f : 0.05f;
            forced_this_bounce = !scout.hit_ground
                              && tau_max >= forced_tau_min
                              && tau_max < FORCED_TAU_CUTOFF;
        }

        uint scatter_shell = 0;

        if (forced_this_bounce) {
            // Upfront forced scattering (unbiased): no analog walk, no double-counting
            float exp_neg_tau = exp(-tau_max);
            weight *= (1.0f - exp_neg_tau);
            if (weight < 1e-30f) break;
            float xi = xorshift_f32(rng);
            float tau_s = -metal_log1p(-xi * (1.0f - exp_neg_tau));
            AdvanceResult adv = advance_to_optical_depth(atm, pos, current_dir, tau_s, wl_idx);
            pos = adv.pos;
            current_dir = adv.dir;
            scatter_shell = adv.shell_idx;
        } else {
            // Analog scatter: standard shell-by-shell free-path walk
            bool scatter_found = false;

            for (uint step = 0; step < 200; step++) {
                float r = length(pos);
                int sidx = shell_index_binary(atm, r);
                if (sidx < 0) break; // exited atmosphere

                uint us = uint(sidx);
                ShellGeom sh = read_shell(atm, us);
                ShellOptics op = read_optics(atm, us, wl_idx);

                if (op.extinction < 1e-20f) {
                    ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
                    if (!bnd.found) break;
                    float3 boundary_pos = pos + current_dir * bnd.dist;
                    boundary_pos = snap_to_radius(boundary_pos, bnd.is_outward ? sh.r_outer : sh.r_inner);
                    float n_from = read_refractive_index(atm, us);
                    uint next_s = bnd.is_outward ? us + 1 : us - 1;
                    float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
                    current_dir = refract_at_boundary(current_dir, boundary_pos, n_from, n_to);
                    pos = radial_nudge(boundary_pos, bnd.is_outward);
                    continue;
                }

                // Exponential transform: modified extinction.
                // Bias axis tilted toward terminator at deep twilight.
                float cos_bias = dot(current_dir, setup.term_axis_dir);
                float sigma = op.extinction;
                float sigma_prime = sigma * (1.0f - setup.alpha_et * cos_bias);
                // If sigma_prime <= 0, ET bias is too strong for this direction.
                // Fall back to unbiased extinction (no bias introduced).
                if (sigma_prime <= 0.0f) sigma_prime = sigma;

                float xi = xorshift_f32(rng);
                float free_path = -log(xi) / sigma_prime;

                ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
                if (!bnd.found) break;

                if (free_path >= bnd.dist) {
                    // Boundary crossing weight correction
                    if (setup.alpha_et > 0.0f) {
                        float et_arg = -setup.alpha_et * sigma * cos_bias * bnd.dist;
                        // Use expm1 form for small arguments to avoid cancellation,
                        // and clamp large arguments to avoid overflow.
                        // exp(x) for |x| < 80 is safe in f32 (~3.4e34).
                        // For |x| >= 80, the weight correction is astronomically
                        // large or small; the chain is effectively dead either way,
                        // so we terminate rather than silently skip the correction
                        // (which would leave the estimator sampling from the wrong
                        // distribution without compensating).
                        if (fabs(et_arg) < 80.0f) {
                            weight *= exp(et_arg);
                        } else {
                            weight = 0.0f; // chain is dead
                        }
                    }
                    if (!isfinite(weight)) break;

                    // Fast path: inward crossing from shell 0 is the ground boundary.
                    if (!bnd.is_outward && us == 0u) {
                        float3 boundary_pos = pos + current_dir * bnd.dist;
                        boundary_pos = snap_to_radius(boundary_pos, sh.r_inner);
                        float3 normal = normalize(boundary_pos);
                        // Ground-bounce NEE: direct solar illumination of ground point
                        float cos_sun_ground = dot(sun_dir, normal);
                        if (cos_sun_ground > 0.0f) {
                            float t_sun_gb = shadow_ray_transmittance(atm, boundary_pos, sun_dir, wl_idx);
                            if (t_sun_gb > 1e-30f) {
                                float albedo_nee = read_albedo(atm, wl_idx);
                                float nee_gb = weight * albedo_nee * t_sun_gb * cos_sun_ground / PI;
                                if (isfinite(nee_gb)) total_I.add(nee_gb);
                            }
                        }
                        float albedo = read_albedo(atm, wl_idx);
                        weight *= albedo;
                        if (!isfinite(weight) || fabs(weight) < 1e-30f) break;
                        prev_dir = current_dir;
                        current_dir = sample_hemisphere(normal, rng);
                        pos = radial_nudge(boundary_pos, true);
                        stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
                        continue;
                    }

                    float3 boundary_pos = pos + current_dir * bnd.dist;
                    boundary_pos = snap_to_radius(boundary_pos, bnd.is_outward ? sh.r_outer : sh.r_inner);

                    // Ground reflection: depolarizes
                    if (!bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                        float3 normal = normalize(boundary_pos);
                        // Ground-bounce NEE
                        float cos_sun_ground = dot(sun_dir, normal);
                        if (cos_sun_ground > 0.0f) {
                            float t_sun_gb = shadow_ray_transmittance(atm, boundary_pos, sun_dir, wl_idx);
                            if (t_sun_gb > 1e-30f) {
                                float albedo_nee = read_albedo(atm, wl_idx);
                                float nee_gb = weight * albedo_nee * t_sun_gb * cos_sun_ground / PI;
                                if (isfinite(nee_gb)) total_I.add(nee_gb);
                            }
                        }
                        float albedo = read_albedo(atm, wl_idx);
                        weight *= albedo;
                        if (!isfinite(weight) || fabs(weight) < 1e-30f) break;
                        prev_dir = current_dir;
                        current_dir = sample_hemisphere(normal, rng);
                        pos = radial_nudge(boundary_pos, true);
                        stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
                        continue;
                    }

                    // Refract and nudge past boundary
                    {
                        float n_from = read_refractive_index(atm, us);
                        uint next_s = bnd.is_outward ? us + 1 : us - 1;
                        float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
                        current_dir = refract_at_boundary(current_dir, boundary_pos, n_from, n_to);
                    }
                    pos = radial_nudge(boundary_pos, bnd.is_outward);
                    continue;
                }

                // Scatter within this shell.
                // Weight correction: (sigma/sigma') * exp(-alpha * sigma * cos_bias * d)
                // If the exp argument would overflow f32, terminate the chain
                // rather than silently skipping the correction (which would leave
                // the estimator sampling from the transformed distribution without
                // the compensating weight, introducing bias).
                if (setup.alpha_et > 0.0f) {
                    float et_arg = -setup.alpha_et * sigma * cos_bias * free_path;
                    if (fabs(et_arg) < 80.0f) {
                        weight *= (sigma / sigma_prime) * exp(et_arg);
                    } else {
                        weight = 0.0f; // chain is dead
                    }
                }
                if (!isfinite(weight)) break;
                pos = pos + current_dir * free_path;
                scatter_shell = us;
                scatter_found = true;
                break;
            }

            if (!scatter_found) break; // chain terminates: escaped atmosphere
        }

        ShellOptics op = read_optics(atm, scatter_shell, wl_idx);

        // Apply SSA BEFORE NEE: the NEE connection is a scattering
        // interaction at this vertex (matches the corrected CPU chains;
        // SSA-after-NEE overestimated each NEE by 1/ssa).
        weight *= op.ssa;

        // NEE: apply Mueller to photon's actual Stokes state
        if (isfinite(weight) && fabs(weight) > 1e-30f) {
            float t_sun_sec = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
            if (t_sun_sec > 1e-30f) {
                float cos_angle_nee = clamp(dot(sun_dir, current_dir), -1.0f, 1.0f);
                float A_nee, B_nee, C_nee;
                stokes_ABC(cos_angle_nee, op, A_nee, B_nee, C_nee);

                float cos2phi_nee, sin2phi_nee;
                scattering_plane_rotation(prev_dir, current_dir, -sun_dir, cos2phi_nee, sin2phi_nee);
                // Guard against NaN from degenerate geometry
                if (!isfinite(cos2phi_nee)) { cos2phi_nee = 1.0f; sin2phi_nee = 0.0f; }

                float4 nee_stokes = scatter_stokes(A_nee, B_nee, C_nee, cos2phi_nee, sin2phi_nee, stokes);

                float scale = weight * t_sun_sec / (4.0f * PI);
                if (isfinite(scale)) {
                    float nee_I = scale * nee_stokes.x;
                    if (isfinite(nee_I)) total_I.add(nee_I);
                    float nee_Q = scale * nee_stokes.y;
                    if (isfinite(nee_Q)) total_Q.add(nee_Q);
                    float nee_U = scale * nee_stokes.z;
                    if (isfinite(nee_U)) total_U.add(nee_U);
                    float nee_V = scale * nee_stokes.w;
                    if (isfinite(nee_V)) total_V.add(nee_V);
                }
            }
        }

        if (!isfinite(weight) || fabs(weight) < 1e-30f) break;

        // Sample new direction
        float cos_theta = clamp(
            (xorshift_f32(rng) < op.rayleigh_fraction)
                ? sample_rayleigh_analytic(xorshift_f32(rng))
                : sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry),
            -1.0f, 1.0f);
        float phi = 2.0f * PI * xorshift_f32(rng);
        float3 new_dir = scatter_direction(current_dir, cos_theta, phi);
        // Guard: if scatter_direction returned zero/NaN, bail
        if (!isfinite(new_dir.x) || (length(new_dir) < 1e-10f)) break;

        // Update Stokes state through this scatter event
        float A_s, B_s, C_s;
        stokes_ABC(cos_theta, op, A_s, B_s, C_s);
        float cos2phi_s, sin2phi_s;
        scattering_plane_rotation(prev_dir, current_dir, new_dir, cos2phi_s, sin2phi_s);
        if (!isfinite(cos2phi_s)) { cos2phi_s = 1.0f; sin2phi_s = 0.0f; }
        stokes = scatter_stokes(A_s, B_s, C_s, cos2phi_s, sin2phi_s, stokes);

        // Normalize by I (importance weighting -- keeps stokes.x = 1)
        if (isfinite(stokes.x) && stokes.x > 1e-30f) {
            float inv_I = 1.0f / stokes.x;
            stokes *= inv_I;
            // Guard against NaN propagation in Q/U/V
            if (!isfinite(stokes.y)) stokes.y = 0.0f;
            if (!isfinite(stokes.z)) stokes.z = 0.0f;
            if (!isfinite(stokes.w)) stokes.w = 0.0f;
        } else {
            stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
        }

        prev_dir = current_dir;
        current_dir = new_dir;
    }

    // Clamp chain return to prevent f32 overflow when accumulated across
    // many chains per thread. Importance weight spikes (from the 3-branch
    // sampling correction at transition SZAs) can produce values >1e20,
    // which overflow when summed across 16+ chains in the v2 stride loop.
    // A clamp at 1e6 is ~10 OOM above typical chain values (~1.0) and
    // introduces negligible bias (affects <0.001% of chains).
    float4 result = float4(total_I.result(), total_Q.result(), total_U.result(), total_V.result());
    // Only filter non-finite values (not bias -- just numerical safety)
    if (!isfinite(result.x)) result.x = 0.0f;
    if (!isfinite(result.y)) result.y = 0.0f;
    if (!isfinite(result.z)) result.z = 0.0f;
    if (!isfinite(result.w)) result.w = 0.0f;
    return result;
}

// ============================================================================
// Kernel 3: hybrid_scatter (REPARALLELIZED)
//
// Old design: 1 thread per wavelength (catastrophic GPU underutilization --
//   only 21 of thousands of cores active, 23x slower than CPU).
//
// New design: 1 THREADGROUP per wavelength, 256 threads per threadgroup.
//   Each thread handles one LOS step with its own secondary ray loop.
//   Per-wavelength reduction via simd_sum() + threadgroup shared memory.
//
// Dispatch: num_wavelengths threadgroups of 256 threads each.
//   wl_idx  = threadgroup_position_in_grid  (which wavelength)
//   step_idx = thread_position_in_threadgroup (which LOS step)
//
// Output: radiance[wl_idx] (f32) -- one value per wavelength.
// ============================================================================

kernel void hybrid_scatter(
    device const float* atm       [[buffer(0)]],
    device const float* params    [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    uint wl_idx    [[threadgroup_position_in_grid]],
    uint step_idx  [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    uint num_wl = atm_num_wavelengths(atm);
    if (wl_idx >= num_wl) return;

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);
    uint secondary_rays = read_secondary_rays(params);

    float toa_radius = atm_toa_radius(atm);
    float surface_radius = atm_surface_radius(atm);

    // ── LOS geometry (all threads compute same values) ──────────────────
    RaySphereHit toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    bool valid_los = toa_hit.hit && toa_hit.t_far > 0.0f;

    uint num_steps = 0;
    float ds = 0.0f;

    if (valid_los) {
        float los_max = toa_hit.t_far;
        RaySphereHit ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
        bool hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < los_max;
        float los_end = hits_ground ? ground_hit.t_near : los_max;
        if (los_end > 0.0f) {
            num_steps = min(HYBRID_LOS_STEPS, uint(los_end / 500.0f) + 20u);
            ds = los_end / float(num_steps);
        }
    }

    // ── Phase 1: Each thread computes extinction*ds for its LOS step ────
    // Store in shared memory for the optical depth prefix computation.

    threadgroup float shared_ext_ds[HYBRID_THREADGROUP_SIZE];

    float my_ext_ds = 0.0f;
    float my_beta_scat = 0.0f;
    float3 scatter_pos = float3(0.0f);
    int my_sidx = -1;
    ShellOptics my_op = {};

    if (valid_los && step_idx < num_steps) {
        float s = (float(step_idx) + 0.5f) * ds;
        scatter_pos = observer_pos + view_dir * s;
        float r = length(scatter_pos);

        if (r <= toa_radius && r >= surface_radius) {
            my_sidx = shell_index_binary(atm, r);
            if (my_sidx >= 0) {
                my_op = read_optics(atm, uint(my_sidx), wl_idx);
                my_ext_ds = my_op.extinction * ds;
                my_beta_scat = my_op.extinction * my_op.ssa;
            }
        }
    }

    float my_cloud_ds = (my_sidx >= 0) ? read_cloud_extinction(atm, uint(my_sidx)) * ds : 0.0f;
    shared_ext_ds[step_idx] = my_ext_ds;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Compute exclusive prefix tau_obs via threadgroup scan ───
    // This replaces the old O(N^2) per-thread summation with a Blelloch scan
    // over shared_ext_ds. step_idx gets tau_obs = sum(shared_ext_ds[0..step_idx-1]).
    // The cloud (delta-scaled scattering) tau is scanned in lockstep so the
    // eye path applies Eddington diffuse transmission for the cloud portion.

    threadgroup float shared_prefix[HYBRID_THREADGROUP_SIZE];
    threadgroup float shared_cloud_prefix[HYBRID_THREADGROUP_SIZE];
    shared_prefix[step_idx] = shared_ext_ds[step_idx];
    shared_cloud_prefix[step_idx] = my_cloud_ds;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Upsweep (reduce) phase
    for (uint offset = 1u; offset < HYBRID_THREADGROUP_SIZE; offset <<= 1u) {
        uint idx = ((step_idx + 1u) * offset * 2u) - 1u;
        if (idx < HYBRID_THREADGROUP_SIZE) {
            shared_prefix[idx] += shared_prefix[idx - offset];
            shared_cloud_prefix[idx] += shared_cloud_prefix[idx - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Convert inclusive reduce tree to exclusive scan
    if (step_idx == HYBRID_THREADGROUP_SIZE - 1u) {
        shared_prefix[step_idx] = 0.0f;
        shared_cloud_prefix[step_idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Downsweep phase
    for (uint offset = HYBRID_THREADGROUP_SIZE >> 1u; offset > 0u; offset >>= 1u) {
        uint idx = ((step_idx + 1u) * offset * 2u) - 1u;
        if (idx < HYBRID_THREADGROUP_SIZE) {
            float t = shared_prefix[idx - offset];
            shared_prefix[idx - offset] = shared_prefix[idx];
            shared_prefix[idx] += t;
            float tc = shared_cloud_prefix[idx - offset];
            shared_cloud_prefix[idx - offset] = shared_cloud_prefix[idx];
            shared_cloud_prefix[idx] += tc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float tau_obs = (step_idx < num_steps) ? shared_prefix[step_idx] : 0.0f;
    float tau_cloud_obs = (step_idx < num_steps) ? shared_cloud_prefix[step_idx] : 0.0f;
    float t_obs = exp(-(tau_obs + my_ext_ds * 0.5f))
                * cloud_diffuse_transmittance(atm, tau_cloud_obs + my_cloud_ds * 0.5f);

    // ── Phase 3: Compute per-step Stokes contribution ─────────────────
    // Full [I,Q,U,V] propagation. Single-scatter NEE applies Mueller to
    // unpolarized sunlight [1,0,0,0]. Secondary chains track Stokes state
    // through all bounces.
    float4 contribution = float4(0.0f);

    if (valid_los && step_idx < num_steps && my_sidx >= 0
        && my_beta_scat > 1e-30f && t_obs > 1e-30f)
    {
        // Per-thread RNG: unique seed for (wavelength, step) pair
        ulong rng = read_rng_seed(params) + ulong(wl_idx);
        rng *= 6364136223846793005ul;
        rng += ulong(step_idx);
        rng *= 2862933555777941757ul;
        rng += 1ul;

        // Order 1: deterministic single-scatter NEE (Stokes)
        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);
        if (t_sun > 1e-30f) {
            float cos_theta_1 = dot(sun_dir, view_dir);
            float A_1, B_1, C_1;
            stokes_ABC(cos_theta_1, my_op, A_1, B_1, C_1);
            float scale_1 = my_beta_scat / (4.0f * PI) * t_sun * t_obs * ds;
            float4 ss_stokes = float4(A_1, B_1, 0.0f, 0.0f);
            contribution += ss_stokes * scale_1;
        }

        // Orders 2+: MC secondary chains (full Stokes propagation)
        if (secondary_rays > 0) {
            float3 local_up = normalize(scatter_pos);
            float cos_sza = dot(sun_dir, local_up);
            float sza_deg = acos(clamp(cos_sza, -1.0f, 1.0f)) * (180.0f / PI);
            BranchParams bp = branch_params_for_sza(sza_deg);
            float sza_t_et = clamp((sza_deg - ZENITH_SZA_START_DEG)
                                   / (ZENITH_SZA_FULL_DEG - ZENITH_SZA_START_DEG), 0.0f, 1.0f);
            SecondarySetup setup;
            setup.local_up = local_up;
            setup.term_axis_dir = terminator_axis(local_up, sun_dir, bp.tilt_rad);
            setup.alpha_p = 1.0f - bp.zenith_frac;
            setup.alpha_z = bp.zenith_frac * (1.0f - bp.term_share);
            setup.alpha_t = bp.zenith_frac * bp.term_share;
            setup.n_zenith = bp.n_zenith;
            setup.m_term = bp.m_term;
            setup.alpha_et = EXP_TRANSFORM_ALPHA_MAX * sza_t_et;
            setup.use_forced = (sza_deg >= ZENITH_SZA_START_DEG) ? 1u : 0u;

            KahanAccum mc_I, mc_Q, mc_U, mc_V;
            for (uint ray = 0; ray < secondary_rays; ray++) {
                float4 chain = trace_secondary_chain(atm, scatter_pos, sun_dir, wl_idx,
                                                      my_op, view_dir, setup,
                                                      ray, secondary_rays, rng);
                mc_I.add(chain.x);
                mc_Q.add(chain.y);
                mc_U.add(chain.z);
                mc_V.add(chain.w);
            }
            float inv_rays = 1.0f / float(secondary_rays);
            float4 mc_avg = float4(mc_I.result(), mc_Q.result(),
                                    mc_U.result(), mc_V.result()) * inv_rays;
            float scale_m = my_beta_scat * t_obs * ds;
            contribution += mc_avg * scale_m;
        }
    }

    // ── Phase 4: Two-level Stokes reduction ─────────────────────────────
    float4 simd_total = simd_sum(contribution);

    threadgroup float4 shared_sums[NUM_SIMD_GROUPS];
    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (step_idx == 0) {
        float4 total = float4(0.0f);
        for (uint i = 0; i < NUM_SIMD_GROUPS; i++) {
            total += shared_sums[i];
        }
        output[wl_idx] = total.x;
    }
}

// ============================================================================
// Kernel 3p: hybrid_los_prefix
//
// Precomputes cumulative LOS optical depth for all (wavelength, step) pairs.
// Dispatch: num_wavelengths threads (1D).
// Output: tau_prefix[wl * HYBRID_LOS_STEPS + step] = cumulative tau to step.
//         Negative value = step beyond LOS end (invalid).
//
// This runs ONCE per geometry and is reused across split-dispatch chunks,
// eliminating the O(steps^2) serial prefix that was in hybrid_scatter_v2.
// ============================================================================

kernel void hybrid_los_prefix(
    device const float* atm         [[buffer(0)]],
    device const float* params      [[buffer(1)]],
    device float*       tau_prefix  [[buffer(2)]],
    uint wl_idx [[thread_position_in_grid]]
) {
    uint num_wl = atm_num_wavelengths(atm);
    if (wl_idx >= num_wl) return;

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);

    float toa_radius = atm_toa_radius(atm);
    float surface_radius = atm_surface_radius(atm);

    RaySphereHit toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    if (!toa_hit.hit || toa_hit.t_far <= 0.0f) {
        for (uint s = 0; s < HYBRID_LOS_STEPS; s++)
            tau_prefix[wl_idx * HYBRID_LOS_STEPS + s] = -1.0f;
        return;
    }

    float los_max = toa_hit.t_far;
    RaySphereHit ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    bool hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < los_max;
    float los_end = hits_ground ? ground_hit.t_near : los_max;

    if (los_end <= 0.0f) {
        for (uint s = 0; s < HYBRID_LOS_STEPS; s++)
            tau_prefix[wl_idx * HYBRID_LOS_STEPS + s] = -1.0f;
        return;
    }

    uint num_steps = min(HYBRID_LOS_STEPS, uint(los_end / 500.0f) + 20u);
    float ds = los_end / float(num_steps);

    // O(num_steps) cumulative scan -- replaces O(num_steps^2) per-step prefix
    float running_tau = 0.0f;
    for (uint step = 0; step < HYBRID_LOS_STEPS; step++) {
        if (step >= num_steps) {
            tau_prefix[wl_idx * HYBRID_LOS_STEPS + step] = -1.0f;
            continue;
        }
        // tau to the START of this step (before this step's contribution)
        tau_prefix[wl_idx * HYBRID_LOS_STEPS + step] = running_tau;

        float s_mid = (float(step) + 0.5f) * ds;
        float3 pos = observer_pos + view_dir * s_mid;
        float r = length(pos);
        if (r <= toa_radius && r >= surface_radius) {
            int sidx = shell_index_binary(atm, r);
            if (sidx >= 0) {
                ShellOptics op = read_optics(atm, uint(sidx), wl_idx);
                running_tau += op.extinction * ds;
            }
        }
    }
}

// ============================================================================
// Kernel 3b: hybrid_scatter_v2 (ray-parallel)
//
// Dispatch: (num_wavelengths, num_steps) threadgroups of 64 threads each.
//   tg_pos.x = wavelength index
//   tg_pos.y = LOS step index
//   thread_in_tg.x = ray index within this step
//
// Each thread traces ceil(secondary_rays / 64) chains. Results are reduced
// via simd_sum + Kahan in shared memory. Output layout:
//   output[wl * HYBRID_LOS_STEPS + step] -- CPU sums across steps.
//
// Buffer 3 (tau_prefix) is precomputed by hybrid_los_prefix and reused
// across split-dispatch chunks (same LOS geometry).
// ============================================================================

kernel void hybrid_scatter_v2(
    device const float* atm         [[buffer(0)]],
    device const float* params      [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    uint3 tg_pos    [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_id    [[simdgroup_index_in_threadgroup]]
) {
    uint wl_idx = tg_pos.x;
    uint step_idx = tg_pos.y;
    uint ray_lane = tid_in_tg.x;
    uint num_wl = atm_num_wavelengths(atm);

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);
    uint secondary_rays = read_secondary_rays(params);

    threadgroup uint shared_valid;
    threadgroup float shared_ds;
    threadgroup float3 shared_scatter_pos;
    threadgroup ShellOptics shared_op;
    threadgroup float shared_beta_scat;
    threadgroup float shared_t_obs;
    threadgroup SecondarySetup shared_setup;

    if (ray_lane == 0) {
        bool valid = (wl_idx < num_wl);
        float toa_radius = atm_toa_radius(atm);
        float surface_radius = atm_surface_radius(atm);
        float ds = 0.0f;
        float3 scatter_pos = float3(0.0f);
        ShellOptics my_op = {};
        float my_beta_scat = 0.0f;
        int my_sidx = -1;
        float t_obs = 0.0f;
        SecondarySetup setup = {};

        if (valid) {
            RaySphereHit toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
            if (!toa_hit.hit || toa_hit.t_far <= 0.0f) {
                valid = false;
            } else {
                float los_max = toa_hit.t_far;
                RaySphereHit ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
                bool hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3f && ground_hit.t_near < los_max;
                float los_end = hits_ground ? ground_hit.t_near : los_max;
                if (los_end <= 0.0f) {
                    valid = false;
                } else {
                    uint num_steps = min(HYBRID_LOS_STEPS, uint(los_end / 500.0f) + 20u);
                    ds = los_end / float(num_steps);
                    if (step_idx >= num_steps) {
                        valid = false;
                    } else {
                        float s = (float(step_idx) + 0.5f) * ds;
                        scatter_pos = observer_pos + view_dir * s;
                        float r = length(scatter_pos);
                        if (r > toa_radius || r < surface_radius) {
                            valid = false;
                        } else {
                            my_sidx = shell_index_binary(atm, r);
                            if (my_sidx < 0) {
                                valid = false;
                            } else {
                                my_op = read_optics(atm, uint(my_sidx), wl_idx);
                                my_beta_scat = my_op.extinction * my_op.ssa;
                                if (my_beta_scat < 1e-30f) {
                                    valid = false;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (valid) {
            float tau_obs = 0.0f;
            float tau_cloud_obs = 0.0f;
            for (uint j = 0; j < step_idx; j++) {
                float sj = (float(j) + 0.5f) * ds;
                float3 pj = observer_pos + view_dir * sj;
                float rj = length(pj);
                if (rj <= toa_radius && rj >= surface_radius) {
                    int sj_idx = shell_index_binary(atm, rj);
                    if (sj_idx >= 0) {
                        ShellOptics oj = read_optics(atm, uint(sj_idx), wl_idx);
                        tau_obs += oj.extinction * ds;
                        tau_cloud_obs += read_cloud_extinction(atm, uint(sj_idx)) * ds;
                    }
                }
            }

            // Single exp is more precise and faster than two multiplied exps.
            // exp(a)*exp(b) = exp(a+b), but the product loses precision
            // when both are small (the f32 multiply rounds twice).
            // Cloud portion: Eddington diffuse (single-representation).
            float my_cloud_ds =
                read_cloud_extinction(atm, uint(my_sidx)) * ds;
            t_obs = exp(-(tau_obs + my_op.extinction * ds * 0.5f))
                  * cloud_diffuse_transmittance(atm, tau_cloud_obs + my_cloud_ds * 0.5f);
            if (t_obs < 1e-30f) {
                valid = false;
            } else if (secondary_rays > 0u) {
                float3 local_up = normalize(scatter_pos);
                float cos_sza = dot(sun_dir, local_up);
                float sza_deg = acos(clamp(cos_sza, -1.0f, 1.0f)) * (180.0f / PI);
                BranchParams bp = branch_params_for_sza(sza_deg);
                float sza_t_et = clamp((sza_deg - ZENITH_SZA_START_DEG)
                                       / (ZENITH_SZA_FULL_DEG - ZENITH_SZA_START_DEG), 0.0f, 1.0f);
                setup.local_up = local_up;
                setup.term_axis_dir = terminator_axis(local_up, sun_dir, bp.tilt_rad);
                setup.alpha_p = 1.0f - bp.zenith_frac;
                setup.alpha_z = bp.zenith_frac * (1.0f - bp.term_share);
                setup.alpha_t = bp.zenith_frac * bp.term_share;
                setup.n_zenith = bp.n_zenith;
                setup.m_term = bp.m_term;
                setup.alpha_et = EXP_TRANSFORM_ALPHA_MAX * sza_t_et;
                setup.use_forced = (sza_deg >= ZENITH_SZA_START_DEG) ? 1u : 0u;
            }
        }

        shared_valid = valid ? 1u : 0u;
        shared_ds = ds;
        shared_scatter_pos = scatter_pos;
        shared_op = my_op;
        shared_beta_scat = my_beta_scat;
        shared_t_obs = t_obs;
        shared_setup = setup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool valid = (shared_valid != 0u);
    float ds = shared_ds;
    float3 scatter_pos = shared_scatter_pos;
    ShellOptics my_op = shared_op;
    float my_beta_scat = shared_beta_scat;
    float t_obs = shared_t_obs;
    SecondarySetup setup = shared_setup;

    float my_contribution = 0.0f;

    // Single-scatter NEE (only thread 0)
    // Multiply by secondary_rays because CPU divides ALL output by secondary_rays.
    // SS is deterministic and must survive that division unchanged.
    if (valid && ray_lane == 0) {
        float t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wl_idx);
        if (t_sun > 1e-30f) {
            float cos_theta_1 = dot(sun_dir, view_dir);
            float A_1, B_1, C_1;
            stokes_ABC(cos_theta_1, my_op, A_1, B_1, C_1);
            float scale_1 = my_beta_scat / (4.0f * PI) * t_sun * t_obs * ds;
            my_contribution += A_1 * scale_1 * float(secondary_rays);
        }
    }

    // Secondary chains: each thread traces a subset of rays
    if (valid && secondary_rays > 0) {
        // ray_offset from upper 32 bits of seed (for split-dispatch)
        ulong raw_seed = read_rng_seed(params);
        uint ray_offset = uint(raw_seed >> 32);
        ulong base_seed = raw_seed & 0xFFFFFFFFul;
        // global_total_rays for stratification (encoded in photons_per_wl)
        uint global_total_rays = read_photons_per_wl(params);
        if (global_total_rays == 0u) global_total_rays = secondary_rays;

        ulong seed_input = base_seed
            ^ (ulong(wl_idx) * 0x9E3779B97F4A7C15ul)
            ^ (ulong(step_idx) << 16)
            ^ (ulong(ray_lane + ray_offset) << 32);
        ulong rng = splitmix64(seed_input);

        // Raw sum without inv_rays -- CPU divides in f64 after reduction
        float scale_m = my_beta_scat * t_obs * ds;
        KahanAccum mc_I;
        for (uint ray = ray_lane; ray < secondary_rays; ray += HYBRID_V2_THREADGROUP_SIZE) {
            uint global_ray = ray + ray_offset;
            float4 chain = trace_secondary_chain(atm, scatter_pos, sun_dir, wl_idx,
                                                  my_op, view_dir, setup,
                                                  global_ray, global_total_rays, rng);
            float val = chain.x * scale_m;
            if (isfinite(val)) {
                mc_I.add(val);
            }
        }
        float mc_result = mc_I.result();
        if (isfinite(mc_result)) {
            my_contribution += mc_result;
        }
        // Debug: track how many non-finite chain values were dropped
        // (Remove after debugging)
        // If mc_result is not finite, the chain produced inf/NaN values
        // that Kahan couldn't save. This means the underlying chain
        // has f32 overflow in weight * nee_stokes, not just in ET.
    }

    // Guard: zero out non-finite contributions
    if (!isfinite(my_contribution)) {
        my_contribution = 0.0f;
    }

    float simd_total = simd_sum(my_contribution);

    threadgroup float shared_vals[HYBRID_V2_NUM_SIMD_GROUPS];
    if (simd_lane == 0) {
        shared_vals[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (ray_lane == 0) {
        KahanAccum final_sum;
        for (uint i = 0; i < HYBRID_V2_NUM_SIMD_GROUPS; i++) {
            float v = shared_vals[i];
            if (isfinite(v)) {
                final_sum.add(v);
            }
        }
        float result = final_sum.result();
        output[wl_idx * HYBRID_LOS_STEPS + step_idx] = isfinite(result) ? result : 0.0f;
    }
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
    float source_rad   = sources[base + 2];

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
    float d = distance_m;

    KahanAccum integral;

    for (uint step = 0; step < altitude_steps; step++) {
        float h = (float(step) + 0.5f) * dh;
        if (h < observer_elevation) continue;

        float r_src_to_scat = sqrt(d * d + h * h);
        float theta_scatter = PI - atan(d / max(h, 1e-6f));

        // Scattering coefficients at this altitude
        float n_rayleigh = rayleigh_tau / H_RAYLEIGH * exp(-h / H_RAYLEIGH);
        float n_aerosol  = aerosol_tau  / H_AEROSOL  * exp(-h / H_AEROSOL);
        float sigma_total = n_rayleigh + n_aerosol;

        // Phase functions
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
        integral.add(di);
    }

    output[tid] = integral.result();
}
