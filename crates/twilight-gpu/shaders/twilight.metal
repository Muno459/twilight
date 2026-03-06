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

constant float PI = 3.14159265358979323846f;
constant float EARTH_RADIUS_M = 6371000.0f;
constant float TOA_ALTITUDE_M = 100000.0f;

constant uint MAX_WAVELENGTHS = 64;
constant uint MAX_LOS_STEPS = 200;
constant uint MAX_SCATTERS = 100;
constant uint HYBRID_LOS_STEPS = 200;
constant uint HYBRID_MAX_BOUNCES = 50;
constant uint HYBRID_THREADGROUP_SIZE = 256;
constant uint SIMD_WIDTH = 32;
constant uint NUM_SIMD_GROUPS = HYBRID_THREADGROUP_SIZE / SIMD_WIDTH; // 8

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

// Garstang constants
constant float H_RAYLEIGH = 8500.0f;
constant float H_AEROSOL  = 1500.0f;
constant float TAU_RAYLEIGH_550 = 0.0962f;

// Boundary nudge distance (meters). Must exceed f32 ULP at Earth radius
// (~0.5m) to guarantee shell crossing after radial nudge.
constant float BOUNDARY_NUDGE_M = 2.0f;

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

    // Minimum distance threshold: must exceed f32 noise at Earth scale.
    // At r ~ 6.4e6, ULP is ~0.5m. We nudge by 2m, so any valid hit
    // should be at least ~shell_thickness away (hundreds of meters).
    // Using 1e-5 (10 um) safely filters self-intersections.
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

    // Fallback: inner sphere only
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

float3 refract_at_boundary(float3 dir, float3 boundary_pos, float n_from, float n_to) {
    // Fast path: no refraction when indices match
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
    // If the scatter point is behind Earth (projection onto sun axis is
    // negative) AND inside the geometric shadow cylinder (perpendicular
    // distance to sun axis < Earth radius), the sun is unreachable.
    // Two dot products, zero accuracy loss, eliminates 50+ shell
    // traversals for points deep in Earth's shadow.
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

    KahanAccum tau;

    // Find initial shell once (O(log N)), then track directly (O(1) per step).
    int sidx = shell_index_binary(atm, length(pos));
    if (sidx < 0) return 1.0f;
    uint us = uint(sidx);

    for (uint iter = 0; iter < 200; iter++) {
        float r_inner = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE];
        float r_outer = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE + 1];

        // Read extinction for this shell and wavelength
        uint optics_idx = us * MAX_WAVELENGTHS + wl_idx;
        float extinction = atm[ATM_OPTICS_START + optics_idx * ATM_OPTICS_STRIDE];

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) break;

        // Accumulate optical depth with Kahan summation
        tau.add(extinction * bnd.dist);

        // Refract at boundary
        float3 boundary_pos = pos + dir * bnd.dist;
        // Snap to exact boundary radius to prevent cumulative f32 position drift.
        float target_r = bnd.is_outward ? r_outer : r_inner;
        float bp_len = length(boundary_pos);
        if (bp_len > 0.0f) {
            boundary_pos *= (target_r / bp_len);
        }
        float n_from = read_refractive_index(atm, us);
        uint next_shell = bnd.is_outward ? us + 1 : us - 1;
        float n_to = (next_shell < ns) ? read_refractive_index(atm, next_shell) : 1.0f;

        dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);

        // Radial nudge past boundary
        pos = radial_nudge(boundary_pos, bnd.is_outward);

        // Ground hit: fully opaque
        if (!bnd.is_outward && length(pos) <= surface_radius + 1.0f) {
            return 0.0f;
        }

        // Exited atmosphere
        if (next_shell >= ns) break;
        us = next_shell;

        if (tau.result() > 50.0f) return 0.0f;
    }

    return exp(-tau.result());
}

// ============================================================================
// Sampling functions
// ============================================================================

inline float sample_rayleigh_analytic(float xi) {
    float q = 8.0f * xi - 4.0f;
    float disc = q * q * 0.25f + 1.0f;
    float sqrt_disc = sqrt(disc);
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

    KahanAccum radiance;
    KahanAccum tau_obs;

    float cos_theta = dot(sun_dir, -view_dir);

    for (uint step = 0; step < num_steps; step++) {
        float s = (float(step) + 0.5f) * ds;
        float3 scatter_pos = observer_pos + view_dir * s;
        float r = length(scatter_pos);

        if (r > toa_radius || r < surface_radius) continue;

        int sidx = shell_index_binary(atm, r);
        if (sidx < 0) continue;

        ShellOptics op = read_optics(atm, uint(sidx), wl_idx);
        float beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30f) {
            tau_obs.add(op.extinction * ds);
            continue;
        }

        // exp(-(A+B)) = exp(-A)*exp(-B) avoids f32 precision loss when
        // adding a small half-step to a large accumulated tau.
        float t_obs = exp(-tau_obs.result()) * exp(-op.extinction * ds * 0.5f);

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

    // Ground reflection (Lambertian BRDF = albedo / pi)
    if (hits_ground) {
        float albedo = read_albedo(atm, wl_idx);
        if (albedo > 1e-10f) {
            float3 ground_pos = observer_pos + view_dir * los_end;
            float3 ground_normal = normalize(ground_pos);
            float cos_sun_incidence = dot(sun_dir, ground_normal);

            if (cos_sun_incidence > 0.0f) {
                float t_sun_ground = shadow_ray_transmittance(atm, ground_pos, sun_dir, wl_idx);
                float t_obs_ground = exp(-tau_obs.result());
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

    float surface_radius = EARTH_RADIUS_M;

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
        float free_path = -log(1.0f - xi + 1e-30f) / op.extinction;

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
            float cos_angle = dot(sun_dir, -dir);
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
                             thread ulong &rng) {
    float3 local_up = normalize(start_pos);
    float surface_radius = EARTH_RADIUS_M;

    // Importance sampling: 50/50 phase-function vs upward-biased
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
        // Compute cos_theta for the initial scatter (for Stokes update)
        cos_theta_init = dot(sun_dir, dir);
    }

    // Initialize Stokes state: apply first scatter's Mueller to incoming state
    // The incoming photon from the LOS single-scatter is effectively [1,0,0,0]
    // at the first secondary bounce. The prev_dir_in -> sun_dir -> dir
    // rotation gives us the initial Stokes transformation.
    float4 stokes = float4(1.0f, 0.0f, 0.0f, 0.0f); // normalized (I=1)

    // Apply initial scatter Mueller to stokes
    {
        float A0, B0, C0;
        stokes_ABC(cos_theta_init, start_optics, A0, B0, C0);
        float cos2phi0, sin2phi0;
        scattering_plane_rotation(prev_dir_in, sun_dir, dir, cos2phi0, sin2phi0);
        stokes = scatter_stokes(A0, B0, C0, cos2phi0, sin2phi0, stokes);
        // Normalize by I (importance weighting)
        if (stokes.x > 1e-30f) {
            float inv_I = 1.0f / stokes.x;
            stokes *= inv_I;
        }
    }

    float3 pos = start_pos;
    float3 current_dir = dir;
    float3 prev_dir = sun_dir; // direction before current propagation segment
    float weight = start_optics.ssa;

    // KBN accumulators for each Stokes component
    KahanAccum total_I, total_Q, total_U, total_V;

    for (uint bounce = 0; bounce < HYBRID_MAX_BOUNCES; bounce++) {
        float r = length(pos);
        int sidx = shell_index_binary(atm, r);
        if (sidx < 0) break;

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

        float xi = xorshift_f32(rng);
        float free_path = -log(1.0f - xi + 1e-30f) / op.extinction;

        ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) break;

        if (free_path >= bnd.dist) {
            float3 boundary_pos = pos + current_dir * bnd.dist;
            boundary_pos = snap_to_radius(boundary_pos, bnd.is_outward ? sh.r_outer : sh.r_inner);

            // Ground reflection: depolarizes
            if (!bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                float albedo = read_albedo(atm, wl_idx);
                weight *= albedo;
                if (weight < 1e-30f) break;
                float3 normal = normalize(boundary_pos);
                prev_dir = current_dir;
                current_dir = sample_hemisphere(normal, rng);
                pos = radial_nudge(boundary_pos, true);
                // Ground reflection depolarizes: reset to unpolarized
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

        // Scatter event
        pos = pos + current_dir * free_path;

        // NEE: apply Mueller to photon's actual Stokes state
        float t_sun_sec = shadow_ray_transmittance(atm, pos, sun_dir, wl_idx);
        if (t_sun_sec > 1e-30f) {
            float cos_angle_nee = dot(sun_dir, -current_dir);
            float A_nee, B_nee, C_nee;
            stokes_ABC(cos_angle_nee, op, A_nee, B_nee, C_nee);

            // Rotation from current propagation plane to NEE (sun) plane
            float cos2phi_nee, sin2phi_nee;
            scattering_plane_rotation(prev_dir, current_dir, -sun_dir, cos2phi_nee, sin2phi_nee);

            // Apply Mueller to photon's Stokes state (NOT to [1,0,0,0])
            float4 nee_stokes = scatter_stokes(A_nee, B_nee, C_nee, cos2phi_nee, sin2phi_nee, stokes);

            float scale = weight * t_sun_sec / (4.0f * PI);
            total_I.add(scale * nee_stokes.x);
            total_Q.add(scale * nee_stokes.y);
            total_U.add(scale * nee_stokes.z);
            total_V.add(scale * nee_stokes.w);
        }

        weight *= op.ssa;
        if (weight < 1e-30f) break;

        // Sample new direction
        float cos_theta;
        if (xorshift_f32(rng) < op.rayleigh_fraction) {
            cos_theta = sample_rayleigh_analytic(xorshift_f32(rng));
        } else {
            cos_theta = sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry);
        }
        float phi = 2.0f * PI * xorshift_f32(rng);
        float3 new_dir = scatter_direction(current_dir, cos_theta, phi);

        // Update Stokes state through this scatter event
        float A_s, B_s, C_s;
        stokes_ABC(cos_theta, op, A_s, B_s, C_s);
        float cos2phi_s, sin2phi_s;
        scattering_plane_rotation(prev_dir, current_dir, new_dir, cos2phi_s, sin2phi_s);
        stokes = scatter_stokes(A_s, B_s, C_s, cos2phi_s, sin2phi_s, stokes);

        // Normalize by I (importance weighting -- keeps stokes.x = 1)
        if (stokes.x > 1e-30f) {
            float inv_I = 1.0f / stokes.x;
            stokes *= inv_I;
        } else {
            // Polarization state degenerate, reset
            stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
        }

        prev_dir = current_dir;
        current_dir = new_dir;
    }

    return float4(total_I.result(), total_Q.result(), total_U.result(), total_V.result());
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

    float toa_radius = EARTH_RADIUS_M + TOA_ALTITUDE_M;
    float surface_radius = EARTH_RADIUS_M;

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

    shared_ext_ds[step_idx] = my_ext_ds;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Compute tau_obs at this step via sequential scan ────────
    // Each thread sums shared_ext_ds[0..step_idx-1]. This is O(N) per thread,
    // O(N^2) total, but N=200 and the cost is negligible compared to the
    // secondary chain tracing (thousands of operations per step).

    float tau_obs = 0.0f;
    for (uint i = 0; i < step_idx && i < num_steps; i++) {
        tau_obs += shared_ext_ds[i];
    }
    // exp(-(A+B)) = exp(-A)*exp(-B) avoids f32 precision loss
    float t_obs = exp(-tau_obs) * exp(-my_ext_ds * 0.5f);

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
            float cos_theta_1 = dot(sun_dir, -view_dir);
            float A_1, B_1, C_1;
            stokes_ABC(cos_theta_1, my_op, A_1, B_1, C_1);

            // Single-scatter from unpolarized sunlight: Mueller * [1,0,0,0]
            // I' = A, Q' = B, U' = 0, V' = 0
            float4 ss_stokes = float4(A_1, B_1, 0.0f, 0.0f);
            float scale_1 = my_beta_scat / (4.0f * PI) * t_sun * t_obs * ds;
            contribution += ss_stokes * scale_1;
        }

        // Orders 2+: MC secondary chains (full Stokes propagation)
        // The noise gate (t_sun > 1e-20) has been REMOVED. The DS discriminant,
        // stable quadratic, KBN summation, and umbra culling provide the
        // precision needed for f32 shadow rays to return correct results at
        // deep twilight without suppressing physics.
        if (secondary_rays > 0) {
            KahanAccum mc_I, mc_Q, mc_U, mc_V;
            for (uint ray = 0; ray < secondary_rays; ray++) {
                float4 chain = trace_secondary_chain(atm, scatter_pos, sun_dir, wl_idx,
                                                      my_op, view_dir, rng);
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
    //
    // Level 1: SIMD reduction on float4 (hardware-accelerated)
    // Level 2: Threadgroup reduction (8 SIMD sums -> 1 final sum)
    //
    // Metal's simd_sum operates on float4 natively (component-wise).

    float4 simd_total = simd_sum(contribution);

    threadgroup float4 shared_sums[NUM_SIMD_GROUPS];

    if (simd_lane == 0) {
        shared_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 writes the final reduced result
    // Output intensity (Stokes I). Full Stokes output can be enabled by
    // expanding the output buffer to 4*num_wl floats.
    if (step_idx == 0) {
        float4 total = float4(0.0f);
        for (uint i = 0; i < NUM_SIMD_GROUPS; i++) {
            total += shared_sums[i];
        }
        output[wl_idx] = total.x;  // Stokes I = total intensity
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
