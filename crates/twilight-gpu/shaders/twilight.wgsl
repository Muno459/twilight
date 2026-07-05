// Twilight MCRT - WGSL compute kernels (portable wgpu backend)
//
// A faithful translation of shaders/twilight.metal (the working Metal
// reference; the CPU f64 engine in twilight-core is the physics
// specification). Six kernels:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_context_prefix     - Deterministic eye-path context (per run)
//   4. hybrid_scatter_v2         - LOS + secondary MC chains (ray-parallel)
//   5. field_tau_probe           - Device-side field DDA probe (G-DDA gates)
//   6. garstang_zenith           - Light pollution skyglow
//
// Buffer layouts match crates/twilight-gpu/src/buffers.rs (v4) exactly and
// byte-for-byte the Metal backend's buffers (shared packing code). The
// offset constants below are kept in sync with buffers.rs by a parse test
// (shader_constants_match_buffers_rs_wgsl), mirroring the Metal test.
//
// TRANSLATION NOTES (deviations from the MSL source, all mechanical):
//
// - RNG / 64-bit integers: WGSL has no u64, so the xorshift64 state and
//   the splitmix64 / LCG seed derivations run on an emulated u64 =
//   vec2<u32>(lo, hi) with exact 64-bit add / multiply / shift / xor.
//   The derivation is BIT-IDENTICAL to the Metal port (same constants,
//   same per-thread seeding by wavelength / step / lane, same
//   (x >> 40) + 1 top-24-bit float conversion), so Metal and wgpu draw
//   the SAME random streams for the same seeds and are directly
//   comparable chain-for-chain (up to f32 branch divergence).
//
// - Reduction: Metal's simd_sum + 2-group Kahan is replaced by the
//   PORTABLE baseline: each of the 64 lanes writes its contribution to
//   workgroup shared memory, one barrier, lane 0 Kahan-sums all 64
//   entries. wgpu subgroups are an optional feature (not universal), and
//   the CPU/GPU comparison is statistical, so reduction-order differences
//   are irrelevant; the baseline runs everywhere.
//
// - isfinite: WGSL has no isfinite() and implementations may assume
//   NaN-free arithmetic, so all finiteness guards use an exact exponent
//   bit test (is_finite_f32), which no fast-math pass can elide.
//
// - INFINITY: WGSL has no infinity literal; the clear-air free path uses
//   FREE_PATH_INF = 1e30 (any in-atmosphere segment is < 1e8 m, so the
//   comparison semantics are unchanged).
//
// - fma(): the DS (double-single) discriminant in ray_sphere_intersect
//   requires a FUSED multiply-add for its error-free transformation. WGSL
//   fma() maps to hardware FMA on Metal / Vulkan / DX12 in practice; a
//   driver that lowers it to unfused mul+add degrades the discriminant to
//   plain f32, which the single-scatter parity gate (1.5e-3) would catch
//   on such hardware. Verified fused on Apple (wgpu-Metal) by the gates.
//
// - round(): WGSL round() is round-half-to-even; Metal/CPU round half
//   away from zero. The substep budget allocation uses round_half_up()
//   (floor(x + 0.5), exact for the non-negative operands there).

// ============================================================================
// Bindings (one shared group; each kernel's auto layout keeps only the
// bindings it statically uses).
//
//   binding 0: atm     - packed atmosphere (PackedAtmosphere)
//   binding 1: params  - packed dispatch params (PackedDispatchParams)
//   binding 2: out_buf - kernel output. For hybrid_context_prefix the host
//                        binds the CONTEXT buffer here (it is that kernel's
//                        output); hybrid_scatter_v2 then reads it at
//                        binding 4.
//   binding 3: fld     - packed 3D cloud field (PackedCloudField; stub when
//                        field_present == 0, never read then)
//   binding 4: hctx    - precomputed eye-path context (read by hybrid v2)
//   binding 5: gsrc    - garstang light sources
//   binding 6: gcfg    - garstang config
//   binding 7: rays    - field_tau_probe ray list
// ============================================================================

@group(0) @binding(0) var<storage, read>       atm:     array<f32>;
@group(0) @binding(1) var<storage, read>       params:  array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(3) var<storage, read>       fld:     array<f32>;
@group(0) @binding(4) var<storage, read>       hctx:    array<f32>;
@group(0) @binding(5) var<storage, read>       gsrc:    array<f32>;
@group(0) @binding(6) var<storage, read>       gcfg:    array<f32>;
@group(0) @binding(7) var<storage, read>       rays:    array<f32>;

// ============================================================================
// Constants (values mirror twilight.metal; ATM_*/FIELD_* offsets mirror
// buffers.rs and are guarded by the offset-sync parse test)
// ============================================================================

const PI: f32 = 3.14159265358979323846;
const INV_4PI: f32 = 1.0 / (4.0 * PI);
const EARTH_RADIUS_M: f32 = 6371008.7714;
const TOA_ALTITUDE_M: f32 = 100000.0;

const MAX_WAVELENGTHS: u32 = 64u;
const MAX_LOS_STEPS: u32 = 200u;
const MAX_SCATTERS: u32 = 100u;
const HYBRID_LOS_STEPS: u32 = 200u;
const HYBRID_MAX_BOUNCES: u32 = 2000u;
const HYBRID_FIELD_MAX_BOUNCES: u32 = 400u;

const HCTX_TAU_OBS: u32 = 0u;
const HCTX_TAU_CLOUD: u32 = 1u;
const HCTX_K_SUB: u32 = 2u;
const HCTX_SPARE: u32 = 3u;
const HCTX_SUB_TAU: u32 = 4u;
const HCTX_SUB_START: u32 = 68u;   // 4 + 64
const HCTX_SUB_COUNT: u32 = 132u;  // 4 + 128
const HCTX_STRIDE: u32 = 196u;     // 4 + 192

const HYBRID_V2_THREADGROUP_SIZE: u32 = 64u;

const CLOUD_SUBSTEP_TAU: f32 = 0.25;
const CLOUD_MAX_SUBSTEPS: u32 = 64u;

const ATM_HEADER_MAGIC: u32 = 0u;
const ATM_HEADER_VERSION: u32 = 1u;
const BUFFER_MAGIC: u32 = 0x544C5754u; // "TWLT"
const BUFFER_VERSION: u32 = 5u;
const HEADER_SENTINEL: f32 = -1.0;
const ATM_NUM_SHELLS: u32 = 2u;
const ATM_NUM_WAVELENGTHS: u32 = 3u;
const ATM_SHELLS_START: u32 = 4u;
const ATM_SHELL_STRIDE: u32 = 4u;
const ATM_OPTICS_START: u32 = 260u;   // 4 + 4*64
const ATM_OPTICS_STRIDE: u32 = 4u;
const ATM_ALBEDO_START: u32 = 16708u;  // 16644 + 64
const ATM_REFRACTIVE_INDEX_START: u32 = 16772u; // 16708 + 64 (v2)
const ATM_CLOUD_EXT_START: u32 = 16836u; // 16772 + 64 (v3)
const ATM_CLOUD_G_SCALED: u32 = 16900u;  // 16836 + 64 (v3)

// Garstang constants
const H_RAYLEIGH: f32 = 8500.0;
const H_AEROSOL: f32 = 1500.0;
const TAU_RAYLEIGH_550: f32 = 0.0962;

const BOUNDARY_NUDGE_M: f32 = 2.0;
const ZENITH_SZA_START_DEG: f32 = 96.0;
const FORCED_TAU_CUTOFF: f32 = 20.0;
const EXP_TRANSFORM_ALPHA_MAX: f32 = 0.5;
const ZENITH_SZA_FULL_DEG: f32 = 106.0;
const ZENITH_BIAS_N: f32 = 5.0;
const ZENITH_MAX_FRACTION: f32 = 0.95;
const TERMINATOR_MAX_SHARE: f32 = 0.5;
const TERMINATOR_N_MAX: f32 = 8.0;
const TERMINATOR_TILT_MIN_DEG: f32 = 20.0;
const TERMINATOR_TILT_MAX_DEG: f32 = 60.0;

// ── Deep-twilight guiding stack (port of the CPU secondary chains; see the
// MSL twin for the full derivation comments) ────────────────────────────────
//
// GPU DIFFERENCE, documented honestly: the CPU keeps
// VSPG_MAX_SEGMENTS = 128 (a full 64-shell double crossing plus reflection
// headroom); per-thread GPU storage is register/stack-limited, so the GPU
// cap is 64 segments = MAX_SHELLS. A full one-way crossing is captured
// exactly; only reflection-multiplied walks overflow, and overflow uses the
// SAME tile-to-tau_max rule as the CPU (extend the LAST segment across the
// overflow tau at neutral importance 1.0): the segment set keeps tiling
// [0, tau_max], so the estimator stays exactly unbiased with merely coarser
// importance on the tail.
const VSPG_GPU_MAX_SEGMENTS: u32 = 64u;
const VSPG_BOOST_START_M: f32 = 15000.0;
const VSPG_BOOST_FULL_M: f32 = 70000.0;
const VSPG_MAX_IMPORTANCE: f32 = 50.0;
const VSPG_SZA_START: f32 = 93.0;
const VSPG_SZA_FULL: f32 = 106.0;

// Dwivedi horizontal direction MIS (CPU constants).
const DWIVEDI_BETA_MAX: f32 = 3.0;
const DWIVEDI_SZA_CENTER: f32 = 103.0;
const DWIVEDI_SZA_WIDTH: f32 = 2.0;
const DWIVEDI_FRAC_MAX: f32 = 0.35;

// Iteration backstop for the field-forced truncated null-collision loop
// (CPU FIELD_NULL_EVENT_LIMIT; Poisson(20) tail, kill = weight zero).
const FIELD_NULL_EVENT_LIMIT: u32 = 512u;

const FLT_MAX_F32: f32 = 3.4028235e38;
// Stand-in for the MSL INFINITY free path (see header notes).
const FREE_PATH_INF: f32 = 1e30;

const DEG_TO_RAD: f32 = PI / 180.0;

// ============================================================================
// Exact bit-level finiteness (WGSL has no isfinite; see header notes)
// ============================================================================

fn is_finite_f32(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7F800000u) != 0x7F800000u;
}

fn is_finite3(v: vec3<f32>) -> bool {
    return is_finite_f32(v.x) && is_finite_f32(v.y) && is_finite_f32(v.z);
}

// Metal round() rounds half away from zero; WGSL round() is half-to-even.
// All call sites pass non-negative operands, where floor(x + 0.5) is the
// half-away-from-zero rounding (matches the CPU and the Metal shader).
fn round_half_up(x: f32) -> f32 {
    return floor(x + 0.5);
}

// copysign(x, y): magnitude of x, sign of y (Metal copysign semantics;
// +0 selects the positive branch, matching copysign's sign-bit read for
// the values that occur here).
fn copysign_f32(x: f32, y: f32) -> f32 {
    return select(abs(x), -abs(x), y < 0.0);
}

// ============================================================================
// Emulated u64 (vec2<u32>: x = lo, y = hi) for the xorshift64 RNG and the
// splitmix64 / LCG seed derivations. Exact 64-bit semantics; constants and
// operation order match twilight.metal bit-for-bit so both GPU backends
// draw identical random streams for identical seeds.
// ============================================================================

// 0x9E3779B97F4A7C15 (golden gamma), 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
const SM64_GAMMA: vec2<u32> = vec2<u32>(0x7F4A7C15u, 0x9E3779B9u);
const SM64_M1: vec2<u32> = vec2<u32>(0x1CE4E5B9u, 0xBF58476Du);
const SM64_M2: vec2<u32> = vec2<u32>(0x133111EBu, 0x94D049BBu);
// 6364136223846793005 = 0x5851F42D4C957F2D, 2862933555777941757 = 0x27BB2EE687B0B0FD
const LCG_K1: vec2<u32> = vec2<u32>(0x4C957F2Du, 0x5851F42Du);
const LCG_K2: vec2<u32> = vec2<u32>(0x87B0B0FDu, 0x27BB2EE6u);

// High 32 bits of a 32x32 -> 64 unsigned multiply (16-bit decomposition).
fn mul32_hi(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    return hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}

// Low 64 bits of a 64x64 multiply.
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x * b.x;
    let hi = mul32_hi(a.x, b.x) + a.x * b.y + a.y * b.x;
    return vec2<u32>(lo, hi);
}

// Shifts for 0 < k < 32 (all call sites use compile-time constants in
// that range; the k = 32 / k = 40 cases are inlined at their call sites).
fn u64_shl(a: vec2<u32>, k: u32) -> vec2<u32> {
    return vec2<u32>(a.x << k, (a.y << k) | (a.x >> (32u - k)));
}

fn u64_shr(a: vec2<u32>, k: u32) -> vec2<u32> {
    return vec2<u32>((a.x >> k) | (a.y << (32u - k)), a.y >> k);
}

fn splitmix64(state: vec2<u32>) -> vec2<u32> {
    var z = u64_add(state, SM64_GAMMA);
    z = u64_mul(z ^ u64_shr(z, 30u), SM64_M1);
    z = u64_mul(z ^ u64_shr(z, 27u), SM64_M2);
    return z ^ u64_shr(z, 31u);
}

// xorshift64 step + top-24-bit conversion to (0, 1]. Identical to the
// Metal xorshift_f32: float((x >> 40) + 1) * (1 / float(2^24 + 1)); note
// float(2^24 + 1) rounds to 2^24 in f32, so the scale is exactly 2^-24.
fn rng_next_f32(state: ptr<function, vec2<u32>>) -> f32 {
    var x = *state;
    x = x ^ u64_shl(x, 13u);
    x = x ^ u64_shr(x, 7u);
    x = x ^ u64_shl(x, 17u);
    *state = x;
    return f32((x.y >> 8u) + 1u) * 5.9604644775390625e-8;
}

// ============================================================================
// Buffer accessor helpers
// ============================================================================

struct ShellGeom {
    r_inner: f32,
    r_outer: f32,
    altitude_mid: f32,
    thickness: f32,
}

struct ShellOptics {
    extinction: f32,
    ssa: f32,
    asymmetry: f32,
    rayleigh_fraction: f32,
}

fn atm_header_valid() -> bool {
    return bitcast<u32>(atm[ATM_HEADER_MAGIC]) == BUFFER_MAGIC
        && bitcast<u32>(atm[ATM_HEADER_VERSION]) == BUFFER_VERSION;
}

fn atm_num_shells() -> u32 {
    return u32(atm[ATM_NUM_SHELLS]);
}

fn atm_num_wavelengths() -> u32 {
    return u32(atm[ATM_NUM_WAVELENGTHS]);
}

fn read_shell(shell_idx: u32) -> ShellGeom {
    let base = ATM_SHELLS_START + shell_idx * ATM_SHELL_STRIDE;
    return ShellGeom(atm[base], atm[base + 1u], atm[base + 2u], atm[base + 3u]);
}

fn read_optics(shell_idx: u32, wl_idx: u32) -> ShellOptics {
    let idx = shell_idx * MAX_WAVELENGTHS + wl_idx;
    let base = ATM_OPTICS_START + idx * ATM_OPTICS_STRIDE;
    return ShellOptics(atm[base], atm[base + 1u], atm[base + 2u], atm[base + 3u]);
}

fn read_albedo(wl_idx: u32) -> f32 {
    return atm[ATM_ALBEDO_START + wl_idx];
}

fn read_refractive_index(shell_idx: u32) -> f32 {
    return atm[ATM_REFRACTIVE_INDEX_START + shell_idx];
}

fn read_cloud_extinction(shell_idx: u32) -> f32 {
    return atm[ATM_CLOUD_EXT_START + shell_idx];
}

// Eddington diffuse transmittance of accumulated (delta-scaled) cloud
// optical depth (single-scatter eye path only; NEVER on chain paths).
fn cloud_diffuse_transmittance(tau_cloud: f32) -> f32 {
    if (tau_cloud <= 0.0) {
        return 1.0;
    }
    let g = atm[ATM_CLOUD_G_SCALED];
    return 1.0 / (1.0 + 0.75 * tau_cloud * (1.0 - g));
}

fn atm_has_cloud_channel(field_present: bool) -> bool {
    if (field_present) {
        return true;
    }
    let ns = atm_num_shells();
    for (var s = 0u; s < ns; s++) {
        if (read_cloud_extinction(s) > 0.0) {
            return true;
        }
    }
    return false;
}

// Dispatch params accessors (PackedDispatchParams layout).
fn read_observer() -> vec3<f32> {
    return vec3<f32>(params[0], params[1], params[2]);
}
fn read_field_present() -> bool {
    return params[3] != 0.0;
}
fn read_view_dir() -> vec3<f32> {
    return vec3<f32>(params[4], params[5], params[6]);
}
fn read_sun_dir() -> vec3<f32> {
    return vec3<f32>(params[8], params[9], params[10]);
}
fn read_photons_per_wl() -> u32 {
    return bitcast<u32>(params[12]);
}
fn read_step_offset() -> u32 {
    return bitcast<u32>(params[7]);
}
fn read_secondary_rays() -> u32 {
    return bitcast<u32>(params[13]);
}
// (lo, hi) of the 64-bit seed.
fn read_rng_seed() -> vec2<u32> {
    return vec2<u32>(bitcast<u32>(params[14]), bitcast<u32>(params[15]));
}

// ============================================================================
// KBN (Kahan-Babuska-Neumaier) compensated summation
// ============================================================================

struct Kahan {
    sum: f32,
    comp: f32,
}

fn kahan_new() -> Kahan {
    return Kahan(0.0, 0.0);
}

fn kahan_add(acc: ptr<function, Kahan>, value: f32) {
    let s = (*acc).sum;
    let t = s + value;
    if (abs(s) >= abs(value)) {
        (*acc).comp += (s - t) + value;
    } else {
        (*acc).comp += (value - t) + s;
    }
    (*acc).sum = t;
}

fn kahan_result(acc: Kahan) -> f32 {
    return acc.sum + acc.comp;
}

// ============================================================================
// log1p polyfill (same construction as the MSL metal_log1p)
// ============================================================================

fn log1p_f32(x: f32) -> f32 {
    if (x > 0.5 || x < -0.5) {
        return log(1.0 + x);
    }
    let u = 1.0 + x;
    let d = u - 1.0; // exact in f32 when |x| <= 0.5
    if (d == 0.0) {
        return x;
    }
    return log(u) * (x / d);
}

// ============================================================================
// Error-free transformations (DS arithmetic via FMA); see the fma() note
// in the header.
// ============================================================================

struct DS {
    hi: f32,
    lo: f32,
}

fn two_product(a: f32, b: f32) -> DS {
    let p = a * b;
    let e = fma(a, b, -p);
    return DS(p, e);
}

fn two_sum(a: f32, b: f32) -> DS {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return DS(s, e);
}

fn ds_add(x: DS, y: DS) -> DS {
    var s = two_sum(x.hi, y.hi);
    s.lo += x.lo + y.lo;
    return two_sum(s.hi, s.lo);
}

fn ds_sub(x: DS, y: DS) -> DS {
    return ds_add(x, DS(-y.hi, -y.lo));
}

// ============================================================================
// Ray-sphere intersection (DS discriminant + stable quadratic)
// ============================================================================

struct RaySphereHit {
    t_near: f32,
    t_far: f32,
    hit: bool,
}

fn ray_sphere_intersect(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> RaySphereHit {
    let a = dot(dir, dir);
    let b_half = dot(origin, dir);
    let r_pos = length(origin);
    let c = (r_pos - radius) * (r_pos + radius);

    let b2 = two_product(b_half, b_half);
    let ac = two_product(a, c);
    let disc_ds = ds_sub(b2, ac);
    let disc = disc_ds.hi + disc_ds.lo;

    var result: RaySphereHit;
    if (disc < 0.0) {
        result.hit = false;
        result.t_near = 0.0;
        result.t_far = 0.0;
        return result;
    }

    let sqrt_disc = sqrt(max(disc, 0.0));
    let q = -(b_half + copysign_f32(sqrt_disc, b_half));

    var t1: f32;
    var t2: f32;
    if (abs(q) > 1e-30) {
        t1 = q / a;
        t2 = c / q;
    } else {
        let inv_a = 1.0 / a;
        t1 = (-b_half - sqrt_disc) * inv_a;
        t2 = (-b_half + sqrt_disc) * inv_a;
    }

    result.t_near = min(t1, t2);
    result.t_far = max(t1, t2);
    result.hit = true;
    return result;
}

// ============================================================================
// Shell index lookup -- O(log N) binary search
// ============================================================================

fn shell_index_binary(r: f32) -> i32 {
    let ns = atm_num_shells();
    if (ns == 0u) {
        return -1;
    }

    let r_inner_first = atm[ATM_SHELLS_START];
    let r_outer_last = atm[ATM_SHELLS_START + (ns - 1u) * ATM_SHELL_STRIDE + 1u];
    if (r < r_inner_first || r >= r_outer_last) {
        return -1;
    }

    var lo = 0u;
    var hi = ns;
    for (var iter = 0u; iter < 32u; iter++) {
        if (lo >= hi) {
            break;
        }
        let mid = lo + (hi - lo) / 2u;
        let r_inner_mid = atm[ATM_SHELLS_START + mid * ATM_SHELL_STRIDE];
        if (r_inner_mid <= r) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo == 0u) {
        return -1;
    }
    return i32(lo - 1u);
}

fn atm_surface_radius() -> f32 {
    let ns = atm_num_shells();
    if (ns == 0u) {
        return EARTH_RADIUS_M;
    }
    return atm[ATM_SHELLS_START];
}

fn atm_toa_radius() -> f32 {
    let ns = atm_num_shells();
    if (ns == 0u) {
        return EARTH_RADIUS_M + TOA_ALTITUDE_M;
    }
    return atm[ATM_SHELLS_START + (ns - 1u) * ATM_SHELL_STRIDE + 1u];
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
    let inv_sqrt_d = inverseSqrt(max(denom, 1e-20));
    return (1.0 - g2) * inv_sqrt_d * inv_sqrt_d * inv_sqrt_d;
}

fn mixed_phase(cos_theta: f32, op: ShellOptics) -> f32 {
    // Exact mixture, ALWAYS (matches the CPU/Metal seed sampler).
    return op.rayleigh_fraction * rayleigh_phase(cos_theta)
        + (1.0 - op.rayleigh_fraction) * henyey_greenstein_phase(cos_theta, op.asymmetry);
}

// ============================================================================
// Stokes [I,Q,U,V] polarized RT helpers
// ============================================================================

fn rayleigh_P12(cos_theta: f32) -> f32 {
    let sin2 = 1.0 - cos_theta * cos_theta;
    return -0.75 * sin2;
}

fn rayleigh_P33(cos_theta: f32) -> f32 {
    return 1.5 * cos_theta;
}

// Trig-free scattering plane rotation (cos 2phi, sin 2phi).
fn scattering_plane_rotation(dir_in: vec3<f32>, dir_out: vec3<f32>, dir_next: vec3<f32>,
                             cos2phi: ptr<function, f32>, sin2phi: ptr<function, f32>) {
    let n1 = cross(dir_in, dir_out);
    let n2 = cross(dir_out, dir_next);

    let n1_sq = dot(n1, n1);
    let n2_sq = dot(n2, n2);

    if (n1_sq < 1e-20 || n2_sq < 1e-20) {
        *cos2phi = 1.0;
        *sin2phi = 0.0;
        return;
    }

    let inv_norm = inverseSqrt(n1_sq * n2_sq);
    var cos_phi = dot(n1, n2) * inv_norm;
    let sin_phi = dot(dir_out, cross(n1, n2)) * inv_norm;

    cos_phi = clamp(cos_phi, -1.0, 1.0);

    *cos2phi = 2.0 * cos_phi * cos_phi - 1.0;
    *sin2phi = 2.0 * sin_phi * cos_phi;
}

fn stokes_ABC(cos_theta: f32, op: ShellOptics,
              A: ptr<function, f32>, B: ptr<function, f32>, C: ptr<function, f32>) {
    let alpha = op.rayleigh_fraction;
    let p11_r = rayleigh_phase(cos_theta);
    let p12_r = rayleigh_P12(cos_theta);
    let p33_r = rayleigh_P33(cos_theta);
    let p11_hg = henyey_greenstein_phase(cos_theta, op.asymmetry);

    *A = alpha * p11_r + (1.0 - alpha) * p11_hg;
    *B = alpha * p12_r;
    *C = alpha * p33_r + (1.0 - alpha) * p11_hg;
}

fn scatter_stokes(A: f32, B: f32, C: f32,
                  cos2phi: f32, sin2phi: f32, s_in: vec4<f32>) -> vec4<f32> {
    let rotQU = cos2phi * s_in.y + sin2phi * s_in.z;
    var s_out: vec4<f32>;
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
    dist: f32,
    is_outward: bool,
    found: bool,
}

fn next_shell_boundary(pos: vec3<f32>, dir: vec3<f32>, r_inner: f32, r_outer: f32) -> ShellBoundary {
    var result: ShellBoundary;
    result.found = false;
    result.dist = 1e30;
    result.is_outward = true;

    let EPS = 1e-5;

    let outer = ray_sphere_intersect(pos, dir, r_outer);
    if (outer.hit) {
        let inner = ray_sphere_intersect(pos, dir, r_inner);

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
            // On-boundary degeneracy (mirrors the CPU fix).
            let m = dot(pos, dir);
            let r2 = dot(pos, pos);
            let b2 = max(r2 - m * m, 0.0);
            if (inner.hit && m < 0.0 && b2 < r_inner * r_inner
                && inner.t_far > EPS
                && abs(length(pos) - r_inner) < 1.0) {
                result.dist = 1e-4;
                result.is_outward = false;
                result.found = true;
                return result;
            }
            result.dist = outer.t_far;
            result.is_outward = true;
            result.found = true;
            return result;
        }

        if (inner.hit && inner.t_near > EPS) {
            result.dist = inner.t_near;
            result.is_outward = false;
            result.found = true;
        }
        return result;
    }

    let inner = ray_sphere_intersect(pos, dir, r_inner);
    if (inner.hit && inner.t_near > EPS) {
        result.dist = inner.t_near;
        result.is_outward = false;
        result.found = true;
    }
    return result;
}

// ============================================================================
// Snell's law refraction at a spherical shell boundary
// ============================================================================

fn refract_at_boundary_r(dir: vec3<f32>, boundary_pos: vec3<f32>, inv_r: f32,
                         n_from: f32, n_to: f32) -> vec3<f32> {
    if (abs(n_from - n_to) < 1e-7) {
        return dir;
    }

    let outward = boundary_pos * inv_r;

    let cos_dir_normal = dot(dir, outward);
    var normal = outward;
    if (cos_dir_normal >= 0.0) {
        normal = -outward;
    }

    let cos_i = -dot(dir, normal);
    let eta = n_from / n_to;
    let k = fma(-eta * eta, fma(-cos_i, cos_i, 1.0), 1.0);

    if (k < 0.0) {
        return fma(normal, vec3<f32>(2.0 * cos_i), dir);
    }

    let cos_t = sqrt(k);
    return fma(normal, vec3<f32>(fma(eta, cos_i, -cos_t)), dir * eta);
}

fn refract_at_boundary(dir: vec3<f32>, boundary_pos: vec3<f32>,
                       n_from: f32, n_to: f32) -> vec3<f32> {
    if (abs(n_from - n_to) < 1e-7) {
        return dir;
    }

    let outward = normalize(boundary_pos);

    let cos_dir_normal = dot(dir, outward);
    var normal = outward;
    if (cos_dir_normal >= 0.0) {
        normal = -outward;
    }

    let cos_i = -dot(dir, normal);
    let eta = n_from / n_to;
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

    if (k < 0.0) {
        return dir + normal * (2.0 * cos_i);
    }

    let cos_t = sqrt(k);
    let factor = eta * cos_i - cos_t;
    return dir * eta + normal * factor;
}

// ============================================================================
// Radial boundary nudge + radius snap
// ============================================================================

fn snap_to_radius(pos: vec3<f32>, target_r: f32) -> vec3<f32> {
    let r = length(pos);
    if (r > 0.0) {
        return pos * (target_r / r);
    }
    return pos;
}

fn radial_nudge(boundary_pos: vec3<f32>, is_outward: bool) -> vec3<f32> {
    let bp_r = length(boundary_pos);
    var radial_dir = vec3<f32>(1.0, 0.0, 0.0);
    if (bp_r > 1e-10) {
        radial_dir = boundary_pos / bp_r;
    }
    let nudge_sign = select(-1.0, 1.0, is_outward);
    return boundary_pos + radial_dir * (nudge_sign * BOUNDARY_NUDGE_M);
}

// ============================================================================
// 3D cloud field (v4): device-side voxel accessor + DDA, a port of
// twilight-core/src/cloud_field.rs via twilight.metal (sigma_at, g_at,
// tau_along, advance_to_tau, next_segment). Header layout mirrors
// buffers.rs::field_offsets.
// ============================================================================

const FIELD_HDR_MAGIC: u32 = 0u;
const FIELD_HDR_VERSION: u32 = 1u;
const FIELD_NZ: u32 = 2u;
const FIELD_NLAT: u32 = 3u;
const FIELD_NLON: u32 = 4u;
const FIELD_TILE: u32 = 5u;
const FIELD_NTLAT: u32 = 6u;
const FIELD_NTLON: u32 = 7u;
const FIELD_G_STAR_PRESENT: u32 = 8u;
const FIELD_BG_PRESENT: u32 = 9u;
const FIELD_MACRO_PRESENT: u32 = 10u;
// v5: per-transport-shell cloud majorant array (host-computed via
// Cloud3DField::band_max_sigma over each shell band; the field-forced
// mode's majorant-combined channel). Present flag + start offset.
const FIELD_SHELL_MAJ_PRESENT: u32 = 11u;
const FIELD_Z0_M: u32 = 12u;
const FIELD_DZ_M: u32 = 13u;
const FIELD_LAT0_DEG: u32 = 14u;
const FIELD_LON0_DEG: u32 = 15u;
const FIELD_DLAT_DEG: u32 = 16u;
const FIELD_DLON_DEG: u32 = 17u;
const FIELD_G_DEFAULT: u32 = 18u;
const FIELD_SHELL_MAJ_OFFSET: u32 = 19u;
const FIELD_SIGMA_OFFSET: u32 = 20u;
const FIELD_G_STAR_OFFSET: u32 = 21u;
const FIELD_MACRO_OFFSET: u32 = 22u;
const FIELD_BG_OFFSET: u32 = 23u;

// core-compatible rem_euclid (WGSL float % is trunc-based like fmod).
fn field_rem_euclid(x: f32, y: f32) -> f32 {
    let r = x % y;
    if (r < 0.0) {
        return r + y;
    }
    return r;
}

struct FieldCoords {
    r: f32,
    lat: f32,
    lon: f32,
}

fn field_sphere_coords(p: vec3<f32>) -> FieldCoords {
    let r = length(p);
    let lat = asin(clamp(p.z / r, -1.0, 1.0)) / DEG_TO_RAD;
    let lon = atan2(p.y, p.x) / DEG_TO_RAD;
    return FieldCoords(r, lat, lon);
}

fn field_uint(slot: u32) -> u32 {
    return u32(fld[slot]);
}

fn field_array_offset(slot: u32) -> u32 {
    return bitcast<u32>(fld[slot]);
}

fn field_z_top_m() -> f32 {
    return fld[FIELD_Z0_M] + fld[FIELD_DZ_M] * f32(field_uint(FIELD_NZ));
}

// True when the v5 per-shell majorant array is packed (the gate for the
// field-forced mode; hosts that packed without an atmosphere leave it off
// and field runs stay analog).
fn field_has_shell_majorants() -> bool {
    return field_uint(FIELD_SHELL_MAJ_PRESENT) == 1u;
}

// Per-transport-shell cloud-extinction majorant (v5): bounds the field's
// sigma_at pointwise over shell `shell_idx`'s radial band. Only valid when
// field_has_shell_majorants().
fn field_shell_majorant(shell_idx: u32) -> f32 {
    return fld[field_array_offset(FIELD_SHELL_MAJ_OFFSET) + shell_idx];
}

// Returns true and writes (iz, ilat, ilon) when inside the footprint.
fn field_indices(r: f32, lat: f32, lon: f32,
                 iz: ptr<function, u32>, ilat: ptr<function, u32>,
                 ilon: ptr<function, u32>) -> bool {
    let z0 = fld[FIELD_Z0_M];
    let dz = fld[FIELD_DZ_M];
    let z = r - EARTH_RADIUS_M;
    if (z < z0 || z >= field_z_top_m()) {
        return false;
    }
    let nz = field_uint(FIELD_NZ);
    let nlat = field_uint(FIELD_NLAT);
    let nlon = field_uint(FIELD_NLON);
    let fiz = (z - z0) / dz;
    let flat_v = (lat - fld[FIELD_LAT0_DEG]) / fld[FIELD_DLAT_DEG];
    let dlon = field_rem_euclid(lon - fld[FIELD_LON0_DEG], 360.0);
    let flon = dlon / fld[FIELD_DLON_DEG];
    if (flat_v < 0.0 || flon < 0.0) {
        return false;
    }
    let ila = u32(flat_v);
    let ilo = u32(flon);
    if (ila >= nlat || ilo >= nlon) {
        return false;
    }
    *iz = min(u32(fiz), nz - 1u);
    *ilat = ila;
    *ilon = ilo;
    return true;
}

// Cloud scattering extinction [1/m] at an ECEF position (sigma_at).
fn field_sigma_at(p: vec3<f32>) -> f32 {
    let c = field_sphere_coords(p);
    var iz = 0u;
    var ilat = 0u;
    var ilon = 0u;
    let nlat = field_uint(FIELD_NLAT);
    let nlon = field_uint(FIELD_NLON);
    if (field_indices(c.r, c.lat, c.lon, &iz, &ilat, &ilon)) {
        let sigma_off = field_array_offset(FIELD_SIGMA_OFFSET);
        return fld[sigma_off + (iz * nlat + ilat) * nlon + ilon];
    }
    // Outside the footprint: background column (or 0 outside z range).
    let z0 = fld[FIELD_Z0_M];
    let dz = fld[FIELD_DZ_M];
    let z = c.r - EARTH_RADIUS_M;
    let nz = field_uint(FIELD_NZ);
    if (z < z0 || z >= field_z_top_m() || field_uint(FIELD_BG_PRESENT) == 0u) {
        return 0.0;
    }
    let iz_bg = min(u32((z - z0) / dz), nz - 1u);
    let bg_off = field_array_offset(FIELD_BG_OFFSET);
    return fld[bg_off + iz_bg];
}

// Asymmetry g* at a position (g_at).
fn field_g_at(p: vec3<f32>) -> f32 {
    if (field_uint(FIELD_G_STAR_PRESENT) == 0u) {
        return fld[FIELD_G_DEFAULT];
    }
    let c = field_sphere_coords(p);
    var iz = 0u;
    var ilat = 0u;
    var ilon = 0u;
    let nlat = field_uint(FIELD_NLAT);
    let nlon = field_uint(FIELD_NLON);
    if (field_indices(c.r, c.lat, c.lon, &iz, &ilat, &ilon)) {
        let g_off = field_array_offset(FIELD_G_STAR_OFFSET);
        return fld[g_off + (iz * nlat + ilat) * nlon + ilon];
    }
    return fld[FIELD_G_DEFAULT];
}

fn field_min_step() -> f32 {
    let dz = fld[FIELD_DZ_M];
    let dxy = fld[FIELD_DLAT_DEG] * DEG_TO_RAD * EARTH_RADIUS_M;
    return min(dz, dxy) * 0.25;
}

// Macrocell majorant for the tile containing p: 0.0 = provably EMPTY tile,
// > 0 = occupied, -1.0 = outside footprint / z range (or no table when
// has_macro is false at the caller). Evaluated at SEGMENT MIDPOINTS only.
fn field_macro_majorant_at(p: vec3<f32>) -> f32 {
    if (field_uint(FIELD_MACRO_PRESENT) == 0u) {
        return -1.0;
    }
    let c = field_sphere_coords(p);
    var iz = 0u;
    var ilat = 0u;
    var ilon = 0u;
    if (!field_indices(c.r, c.lat, c.lon, &iz, &ilat, &ilon)) {
        return -1.0;
    }
    let tile = field_uint(FIELD_TILE);
    let ntlat = field_uint(FIELD_NTLAT);
    let ntlon = field_uint(FIELD_NTLON);
    let off = field_array_offset(FIELD_MACRO_OFFSET);
    let itlat = ilat / tile;
    let itlon = ilon / tile;
    return fld[off + (iz * ntlat + itlat) * ntlon + itlon];
}

// Distance along dir from p to the nearest crossing of one of the SIX
// bounding surfaces of the voxel grid (footprint cap for coarse steps).
fn field_distance_to_footprint_boundary(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    var best = FLT_MAX_F32;
    let z0 = fld[FIELD_Z0_M];
    let lat0 = fld[FIELD_LAT0_DEG];
    let lon0 = fld[FIELD_LON0_DEG];
    let nlat = f32(field_uint(FIELD_NLAT));
    let nlon = f32(field_uint(FIELD_NLON));
    let dlat = fld[FIELD_DLAT_DEG];
    let dlon = fld[FIELD_DLON_DEG];

    // Spheres at z0 and z_top.
    let r = length(p);
    let b_r = dot(p, dir);
    var zs = array<f32, 2>(z0, field_z_top_m());
    for (var i = 0u; i < 2u; i++) {
        let rk = EARTH_RADIUS_M + zs[i];
        let cc = r * r - rk * rk;
        let disc = b_r * b_r - cc;
        if (disc >= 0.0) {
            let s = sqrt(disc);
            let t0 = -b_r - s;
            let t1 = -b_r + s;
            if (t0 > 1e-6 && t0 < best) { best = t0; }
            if (t1 > 1e-6 && t1 < best) { best = t1; }
        }
    }
    // Latitude cones at lat0 and lat0 + nlat*dlat.
    var lats = array<f32, 2>(lat0, lat0 + nlat * dlat);
    for (var i = 0u; i < 2u; i++) {
        let phi = lats[i] * DEG_TO_RAD;
        let tp = tan(phi);
        let a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
        let bq = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
        let cc = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
        if (abs(a) > 1e-30) {
            let disc = bq * bq - a * cc;
            if (disc >= 0.0) {
                let s = sqrt(disc);
                var roots = array<f32, 2>((-bq - s) / a, (-bq + s) / a);
                for (var j = 0u; j < 2u; j++) {
                    let t = roots[j];
                    if (t > 1e-6 && t < best) {
                        // Reject the mirror cone (opposite hemisphere).
                        let zc = p.z + t * dir.z;
                        if (zc * phi >= -1e-9) { best = t; }
                    }
                }
            }
        } else if (abs(bq) > 1e-30) {
            let t = -cc / (2.0 * bq);
            if (t > 1e-6 && t < best) { best = t; }
        }
    }
    // Meridian planes at lon0 and lon0 + nlon*dlon.
    var lons = array<f32, 2>(lon0, lon0 + nlon * dlon);
    for (var i = 0u; i < 2u; i++) {
        let lam = lons[i] * DEG_TO_RAD;
        let sl = sin(lam);
        let cl = cos(lam);
        let denom = dir.x * sl - dir.y * cl;
        if (abs(denom) > 1e-30) {
            let t = -(p.x * sl - p.y * cl) / denom;
            if (t > 1e-6 && t < best) { best = t; }
        }
    }
    return best;
}

// Distance to the nearest COARSE (macro-tile) boundary.
fn field_distance_to_next_tile_boundary(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let c = field_sphere_coords(p);
    let r = c.r;
    var best = FLT_MAX_F32;

    let z0 = fld[FIELD_Z0_M];
    let dz = fld[FIELD_DZ_M];
    let lat0 = fld[FIELD_LAT0_DEG];
    let lon0 = fld[FIELD_LON0_DEG];
    let tile = f32(field_uint(FIELD_TILE));
    let dlat_t = fld[FIELD_DLAT_DEG] * tile;
    let dlon_t = fld[FIELD_DLON_DEG] * tile;

    // Radial (fine z-grid).
    let z = r - EARTH_RADIUS_M;
    let iz = floor((z - z0) / dz);
    var ks_r = array<f32, 4>(iz - 1.0, iz, iz + 1.0, iz + 2.0);
    let b_r = dot(p, dir);
    for (var i = 0u; i < 4u; i++) {
        let rk = EARTH_RADIUS_M + z0 + ks_r[i] * dz;
        let cc = r * r - rk * rk;
        let disc = b_r * b_r - cc;
        if (disc >= 0.0) {
            let s = sqrt(disc);
            let t0 = -b_r - s;
            let t1 = -b_r + s;
            if (t0 > 1e-6 && t0 < best) { best = t0; }
            if (t1 > 1e-6 && t1 < best) { best = t1; }
        }
    }
    // Latitude cones at tile spacing.
    let flat_v = (c.lat - lat0) / dlat_t;
    let kf = floor(flat_v);
    var ks_lat = array<f32, 4>(kf - 1.0, kf, kf + 1.0, kf + 2.0);
    for (var i = 0u; i < 4u; i++) {
        let phi = (lat0 + ks_lat[i] * dlat_t) * DEG_TO_RAD;
        let tp = tan(phi);
        let a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
        let b = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
        let cc = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
        if (abs(a) > 1e-30) {
            let disc = b * b - a * cc;
            if (disc >= 0.0) {
                let s = sqrt(disc);
                var roots = array<f32, 2>((-b - s) / a, (-b + s) / a);
                for (var j = 0u; j < 2u; j++) {
                    let t = roots[j];
                    if (t > 1e-6 && t < best) {
                        let zc = p.z + t * dir.z;
                        if (zc * phi >= -1e-9) { best = t; }
                    }
                }
            }
        } else if (abs(b) > 1e-30) {
            let t = -cc / (2.0 * b);
            if (t > 1e-6 && t < best) { best = t; }
        }
    }
    // Longitude planes at tile spacing.
    let flon = field_rem_euclid(c.lon - lon0, 360.0) / dlon_t;
    let kn = floor(flon);
    var ks_lon = array<f32, 4>(kn - 1.0, kn, kn + 1.0, kn + 2.0);
    for (var i = 0u; i < 4u; i++) {
        let lam = (lon0 + ks_lon[i] * dlon_t) * DEG_TO_RAD;
        let sl = sin(lam);
        let cl = cos(lam);
        let denom = dir.x * sl - dir.y * cl;
        if (abs(denom) > 1e-30) {
            let t = -(p.x * sl - p.y * cl) / denom;
            if (t > 1e-6 && t < best) { best = t; }
        }
    }
    return best;
}

// Distance to the nearest FINE grid-cell boundary (radial shells, latitude
// cones, longitude planes) with the floor-1..floor+2 candidate window.
fn field_distance_to_next_boundary(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let c = field_sphere_coords(p);
    let r = c.r;
    var best = FLT_MAX_F32;

    let z0 = fld[FIELD_Z0_M];
    let dz = fld[FIELD_DZ_M];
    let lat0 = fld[FIELD_LAT0_DEG];
    let dlat = fld[FIELD_DLAT_DEG];
    let lon0 = fld[FIELD_LON0_DEG];
    let dlon = fld[FIELD_DLON_DEG];

    // Radial (sphere) crossings.
    let z = r - EARTH_RADIUS_M;
    let iz = floor((z - z0) / dz);
    var ks_r = array<f32, 4>(iz - 1.0, iz, iz + 1.0, iz + 2.0);
    let b_r = dot(p, dir);
    for (var i = 0u; i < 4u; i++) {
        let rk = EARTH_RADIUS_M + z0 + ks_r[i] * dz;
        let cc = r * r - rk * rk;
        let disc = b_r * b_r - cc;
        if (disc >= 0.0) {
            let s = sqrt(disc);
            let t0 = -b_r - s;
            let t1 = -b_r + s;
            if (t0 > 1e-6 && t0 < best) { best = t0; }
            if (t1 > 1e-6 && t1 < best) { best = t1; }
        }
    }

    // Latitude (cone) crossings.
    let flat_v = (c.lat - lat0) / dlat;
    let kf = floor(flat_v);
    var ks_lat = array<f32, 4>(kf - 1.0, kf, kf + 1.0, kf + 2.0);
    for (var i = 0u; i < 4u; i++) {
        let phi = (lat0 + ks_lat[i] * dlat) * DEG_TO_RAD;
        let tp = tan(phi);
        let a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
        let b = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
        let cc = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
        if (abs(a) > 1e-30) {
            let disc = b * b - a * cc;
            if (disc >= 0.0) {
                let s = sqrt(disc);
                var roots = array<f32, 2>((-b - s) / a, (-b + s) / a);
                for (var j = 0u; j < 2u; j++) {
                    let t = roots[j];
                    if (t > 1e-6 && t < best) {
                        let zc = p.z + t * dir.z;
                        if (zc * phi >= -1e-9) { best = t; }
                    }
                }
            }
        } else if (abs(b) > 1e-30) {
            let t = -cc / (2.0 * b);
            if (t > 1e-6 && t < best) { best = t; }
        }
    }

    // Longitude (meridian plane) crossings.
    let flon = field_rem_euclid(c.lon - lon0, 360.0) / dlon;
    let kn = floor(flon);
    var ks_lon = array<f32, 4>(kn - 1.0, kn, kn + 1.0, kn + 2.0);
    for (var i = 0u; i < 4u; i++) {
        let lam = (lon0 + ks_lon[i] * dlon) * DEG_TO_RAD;
        let sl = sin(lam);
        let cl = cos(lam);
        let denom = dir.x * sl - dir.y * cl;
        if (abs(denom) > 1e-30) {
            let t = -(p.x * sl - p.y * cl) / denom;
            if (t > 1e-6 && t < best) { best = t; }
        }
    }

    return best;
}

// One traversal segment starting at parameter t: writes the constant sigma
// over [t, t_next] to sigma_out and returns t_next. See the CPU
// next_segment for the full derivation; the midpoint-classification and
// footprint-capping rules are preserved exactly.
fn field_next_segment(p0: vec3<f32>, dir: vec3<f32>,
                      t: f32, t_max: f32, min_step: f32,
                      has_macro: bool, sigma_out: ptr<function, f32>) -> f32 {
    let p = p0 + dir * t;
    let d_fine = max(field_distance_to_next_boundary(p, dir), min_step);
    let t_fine = min(t + d_fine, t_max);
    let mid_fine = p0 + dir * ((t + t_fine) * 0.5);
    if (!has_macro) {
        *sigma_out = field_sigma_at(mid_fine);
        return t_fine;
    }
    let maj_f = field_macro_majorant_at(mid_fine);
    if (maj_f > 0.0) {
        // Occupied tile: integrate finely within it.
        *sigma_out = field_sigma_at(mid_fine);
        return t_fine;
    }
    // Empty tile or outside the footprint: coarse extension, capped by the
    // footprint surfaces.
    let d_fp = field_distance_to_footprint_boundary(p, dir);
    let d_coarse = max(min(field_distance_to_next_tile_boundary(p, dir), d_fp), min_step);
    let t_coarse = min(t + d_coarse, t_max);
    let mid_coarse = p0 + dir * ((t + t_coarse) * 0.5);
    let maj_c = field_macro_majorant_at(mid_coarse);
    if (maj_f == 0.0 && maj_c == 0.0) {
        // Provably empty tile: cross it in one step, tau += 0.
        *sigma_out = 0.0;
        return t_coarse;
    }
    if (maj_f < 0.0 && maj_c < 0.0) {
        // Outside the footprint (or z range): sigma is altitude-only and
        // constant over the capped coarse segment.
        *sigma_out = field_sigma_at(mid_coarse);
        return t_coarse;
    }
    // Fine/coarse classification disagreement: fall back to the fine
    // segment with its own midpoint, always valid.
    *sigma_out = field_sigma_at(mid_fine);
    return t_fine;
}

// Exact cloud optical depth along p0 + t*dir, t in [0, t_max].
fn field_tau_along(p0: vec3<f32>, dir: vec3<f32>, t_max: f32) -> f32 {
    if (t_max <= 0.0) {
        return 0.0;
    }
    var tau = 0.0;
    var t = 0.0;
    let min_step = field_min_step();
    let has_macro = field_uint(FIELD_MACRO_PRESENT) != 0u;
    for (var iter = 0u; iter < 40000u; iter++) {
        if (t >= t_max) {
            break;
        }
        var sigma = 0.0;
        let t_next = field_next_segment(p0, dir, t, t_max, min_step, has_macro, &sigma);
        tau += sigma * (t_next - t);
        t = t_next;
    }
    return tau;
}

// Inverse of field_tau_along: parameter t where the accumulated cloud tau
// reaches tau_target, or a negative sentinel if the segment ends first.
fn field_advance_to_tau(p0: vec3<f32>, dir: vec3<f32>,
                        t_max: f32, tau_target: f32) -> f32 {
    if (tau_target <= 0.0) {
        return 0.0;
    }
    var tau = 0.0;
    var t = 0.0;
    let min_step = field_min_step();
    let has_macro = field_uint(FIELD_MACRO_PRESENT) != 0u;
    for (var iter = 0u; iter < 40000u; iter++) {
        if (t >= t_max) {
            return -1.0;
        }
        var sigma = 0.0;
        let t_next = field_next_segment(p0, dir, t, t_max, min_step, has_macro, &sigma);
        let dtau = sigma * (t_next - t);
        if (tau + dtau >= tau_target) {
            // Constant sigma within the segment: linear inversion.
            return t + (tau_target - tau) / max(sigma, 1e-30);
        }
        tau += dtau;
        t = t_next;
    }
    return -1.0;
}

// ============================================================================
// Shadow ray transmittance -- shell-by-shell with refraction
// ============================================================================

fn shadow_ray_transmittance(start_pos: vec3<f32>, sun_dir: vec3<f32>, wl_idx: u32) -> f32 {
    let ns = atm_num_shells();
    let surface_radius = atm[ATM_SHELLS_START]; // r_inner of shell 0

    // Umbra cylinder culling (O(1) pre-check).
    let p_proj = dot(start_pos, sun_dir);
    if (p_proj < 0.0) {
        let cross_ps = cross(start_pos, sun_dir);
        let perp_dist_sq = dot(cross_ps, cross_ps);
        if (perp_dist_sq < surface_radius * surface_radius) {
            return 0.0;
        }
    }

    var pos = start_pos;
    var dir = sun_dir;

    var tau = 0.0;
    var tau_cloud = 0.0;

    let sidx = shell_index_binary(length(pos));
    if (sidx < 0) {
        return 1.0;
    }
    var us = u32(sidx);

    for (var iter = 0u; iter < 200u; iter++) {
        let shell_base = ATM_SHELLS_START + us * ATM_SHELL_STRIDE;
        let r_inner = atm[shell_base];
        let r_outer = atm[shell_base + 1u];

        let extinction = atm[ATM_OPTICS_START + (us * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];

        let bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) {
            break;
        }

        tau += extinction * bnd.dist;
        tau_cloud += read_cloud_extinction(us) * bnd.dist;

        // Snap + nudge.
        var boundary_pos = pos + dir * bnd.dist;
        let target_r = select(r_inner, r_outer, bnd.is_outward);
        let bp_r = length(boundary_pos);
        if (bp_r > 0.0) {
            boundary_pos *= (target_r / bp_r);
        }

        let inv_target_r = 1.0 / target_r;
        let n_from = read_refractive_index(us);
        let next_shell = select(us - 1u, us + 1u, bnd.is_outward);
        var n_to = 1.0;
        if (next_shell < ns) {
            n_to = read_refractive_index(next_shell);
        }
        dir = refract_at_boundary_r(dir, boundary_pos, inv_target_r, n_from, n_to);

        let radial = boundary_pos * inv_target_r;
        let nudge_sign = select(-1.0, 1.0, bnd.is_outward);
        pos = boundary_pos + radial * (nudge_sign * BOUNDARY_NUDGE_M);

        if (!bnd.is_outward && target_r <= surface_radius + 1.0) {
            return 0.0;
        }

        if (next_shell >= ns) {
            break;
        }
        us = next_shell;

        if (tau > 50.0) {
            return 0.0;
        }
    }

    // Clear-air Beer-Lambert x Eddington diffuse for the cloud portion.
    return exp(-tau) * cloud_diffuse_transmittance(tau_cloud);
}

// Chain-mode shadow ray: Beer-Lambert exp(-tau_cloud) with the cloud tau
// integrated by the field DDA (field bound) or per-shell 1D extinction.
// NO T_diff anywhere on chain paths.
fn shadow_ray_transmittance_chain(field_present: bool, cloud_channel: bool,
                                  start_pos: vec3<f32>, sun_dir: vec3<f32>,
                                  wl_idx: u32) -> f32 {
    if (!cloud_channel) {
        return shadow_ray_transmittance(start_pos, sun_dir, wl_idx);
    }
    let ns = atm_num_shells();
    let surface_radius = atm[ATM_SHELLS_START];

    let p_proj = dot(start_pos, sun_dir);
    if (p_proj < 0.0) {
        let cross_ps = cross(start_pos, sun_dir);
        let perp_dist_sq = dot(cross_ps, cross_ps);
        if (perp_dist_sq < surface_radius * surface_radius) {
            return 0.0;
        }
    }

    var pos = start_pos;
    var dir = sun_dir;
    var tau = 0.0;
    var tau_cloud = 0.0;

    let sidx = shell_index_binary(length(pos));
    if (sidx < 0) {
        return 1.0;
    }
    var us = u32(sidx);

    for (var iter = 0u; iter < 200u; iter++) {
        let shell_base = ATM_SHELLS_START + us * ATM_SHELL_STRIDE;
        let r_inner = atm[shell_base];
        let r_outer = atm[shell_base + 1u];
        let extinction = atm[ATM_OPTICS_START + (us * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];

        let bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) {
            break;
        }

        tau += extinction * bnd.dist;
        // Combined opacity early-out BEFORE the per-shell field DDA
        // (identical threshold to the CPU trace_transmittance).
        if (tau + tau_cloud > 35.0) {
            return 0.0;
        }
        // Cloud tau for this straight segment: pos/dir BEFORE refraction.
        if (field_present) {
            tau_cloud += field_tau_along(pos, dir, bnd.dist);
        } else {
            tau_cloud += read_cloud_extinction(us) * bnd.dist;
        }

        var boundary_pos = pos + dir * bnd.dist;
        let target_r = select(r_inner, r_outer, bnd.is_outward);
        let bp_r = length(boundary_pos);
        if (bp_r > 0.0) {
            boundary_pos *= (target_r / bp_r);
        }

        let inv_target_r = 1.0 / target_r;
        let n_from = read_refractive_index(us);
        let next_shell = select(us - 1u, us + 1u, bnd.is_outward);
        var n_to = 1.0;
        if (next_shell < ns) {
            n_to = read_refractive_index(next_shell);
        }
        dir = refract_at_boundary_r(dir, boundary_pos, inv_target_r, n_from, n_to);

        let radial = boundary_pos * inv_target_r;
        let nudge_sign = select(-1.0, 1.0, bnd.is_outward);
        pos = boundary_pos + radial * (nudge_sign * BOUNDARY_NUDGE_M);

        if (!bnd.is_outward && target_r <= surface_radius + 1.0) {
            return 0.0;
        }
        if (next_shell >= ns) {
            break;
        }
        us = next_shell;
        if (tau > 50.0) {
            return 0.0;
        }
    }

    // Clear-air Beer-Lambert x cloud Beer-Lambert (explicit scattering).
    return exp(-tau) * exp(-tau_cloud);
}

// Total path length of the ray inside the spherical shell annulus
// [r_inner, r_outer], t in [0, t_max]. Analytic, no stepping.
fn ray_path_through_shell(origin: vec3<f32>, dir: vec3<f32>,
                          r_inner: f32, r_outer: f32, t_max: f32) -> f32 {
    let outer = ray_sphere_intersect(origin, dir, r_outer);
    if (!outer.hit) {
        return 0.0;
    }
    let o0 = max(outer.t_near, 0.0);
    let o1 = min(outer.t_far, t_max);
    if (o1 <= o0 + 1e-6) {
        return 0.0;
    }

    let inner = ray_sphere_intersect(origin, dir, r_inner);
    var i0 = 0.0;
    var i1 = 0.0;
    var has_inner = false;
    if (inner.hit) {
        i0 = max(inner.t_near, 0.0);
        i1 = min(inner.t_far, t_max);
        has_inner = (i1 > i0 + 1e-6);
    }
    if (!has_inner) {
        return o1 - o0;
    }

    // Shell interval = outer interval minus inner interval.
    var total = 0.0;
    let seg1_end = min(o1, i0);
    if (seg1_end > o0) {
        total += seg1_end - o0;
    }
    let seg2_start = max(o0, i1);
    if (o1 > seg2_start) {
        total += o1 - seg2_start;
    }
    return total;
}

// Exact cloud optical depth of one straight eye-path (sub)step: the shared
// field DDA when a field is bound, else analytic per-shell path lengths
// through the 1D shell deck (CPU eye_step_cloud_tau).
fn eye_step_cloud_tau(field_present: bool, start: vec3<f32>, dir: vec3<f32>, ds: f32) -> f32 {
    if (field_present) {
        return field_tau_along(start, dir, ds);
    }
    var tau = 0.0;
    let ns = atm_num_shells();
    for (var s = 0u; s < ns; s++) {
        let sigma_c = read_cloud_extinction(s);
        if (sigma_c <= 0.0) {
            continue;
        }
        let shell_base = ATM_SHELLS_START + s * ATM_SHELL_STRIDE;
        tau += sigma_c
            * ray_path_through_shell(start, dir, atm[shell_base], atm[shell_base + 1u], ds);
    }
    return tau;
}

// ============================================================================
// Sampling functions
// ============================================================================

// Newton-Raphson cube root from a bit-hack seed (2 Halley iterations).
fn fast_cbrt(a: f32) -> f32 {
    let x = abs(a);
    if (x < 1e-30) {
        return 0.0;
    }
    var y = bitcast<f32>((bitcast<u32>(x) / 3u) + 0x2a508bdbu);
    y = y * (2.0 / 3.0) + x / (3.0 * y * y);
    y = y * (2.0 / 3.0) + x / (3.0 * y * y);
    return copysign_f32(y, a);
}

fn sample_rayleigh_analytic(xi: f32) -> f32 {
    let q = 8.0 * xi - 4.0;
    let disc = fma(q * q, 0.25, 1.0);
    let sqrt_disc = sqrt(disc);
    let a_val = -q * 0.5 + sqrt_disc;
    let b_val = -q * 0.5 - sqrt_disc;
    let mu = fast_cbrt(a_val) + fast_cbrt(b_val);
    return clamp(mu, -1.0, 1.0);
}

fn sample_henyey_greenstein(xi: f32, g: f32) -> f32 {
    if (abs(g) < 1e-6) {
        return 2.0 * xi - 1.0;
    }
    let g2 = g * g;
    let s = (1.0 - g2) / (1.0 - g + 2.0 * g * xi);
    let mu = (1.0 + g2 - s * s) / (2.0 * g);
    return clamp(mu, -1.0, 1.0);
}

fn scatter_direction(dir: vec3<f32>, cos_theta: f32, phi: f32) -> vec3<f32> {
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let cos_phi = cos(phi);
    let sin_phi = sin(phi);

    let w = dir;
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(w.z) >= 0.9) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let u_vec = normalize(cross(w, up));
    let v_vec = cross(w, u_vec);

    // (u_vec, v_vec, w) is orthonormal: result is unit length.
    let sc = sin_theta * cos_phi;
    let ss = sin_theta * sin_phi;
    return sc * u_vec + ss * v_vec + cos_theta * w;
}

fn sample_hemisphere(normal: vec3<f32>, rng: ptr<function, vec2<u32>>) -> vec3<f32> {
    let xi1 = rng_next_f32(rng);
    let xi2 = rng_next_f32(rng);
    let cos_theta = sqrt(xi1);
    let phi = 2.0 * PI * xi2;
    return scatter_direction(normal, cos_theta, phi);
}

struct ZenithSample {
    dir: vec3<f32>,
    cos_theta: f32,
}

// Power-cosine lobe about `normal`; consumes exactly 2 RNG draws.
fn sample_zenith_biased(normal: vec3<f32>, n: f32, rng: ptr<function, vec2<u32>>) -> ZenithSample {
    let xi1 = rng_next_f32(rng);
    let xi2 = rng_next_f32(rng);
    let cos_theta = pow(xi1, 1.0 / (n + 1.0));
    let phi = 2.0 * PI * xi2;
    let dir = scatter_direction(normal, cos_theta, phi);
    return ZenithSample(dir, cos_theta);
}

// PDF over solid angle of the (untruncated) power-cosine lobe.
fn power_cos_pdf(cos_theta: f32, n: f32) -> f32 {
    if (cos_theta <= 0.0) {
        return 0.0;
    }
    return (n + 1.0) * pow(cos_theta, n) / (2.0 * PI);
}

// Density of the 3-component seed mixture at omega.
fn seed_mixture_pdf(omega: vec3<f32>, sun_dir: vec3<f32>, local_up: vec3<f32>,
                    term_axis: vec3<f32>, alpha_p: f32, alpha_z: f32,
                    alpha_t: f32, n_zenith: f32, m_term: f32,
                    op: ShellOptics) -> f32 {
    var q = alpha_p * mixed_phase(dot(omega, sun_dir), op) * INV_4PI;
    if (alpha_z > 1e-6) {
        q += alpha_z * power_cos_pdf(dot(omega, local_up), n_zenith);
    }
    if (alpha_t > 1e-6) {
        q += alpha_t * power_cos_pdf(dot(omega, term_axis), m_term);
    }
    return q;
}

struct BranchParams {
    zenith_frac: f32,
    n_zenith: f32,
    term_share: f32,
    m_term: f32,
    tilt_rad: f32,
}

struct SecondarySetup {
    local_up: vec3<f32>,
    term_axis_dir: vec3<f32>,
    alpha_p: f32,
    alpha_z: f32,
    alpha_t: f32,
    n_zenith: f32,
    m_term: f32,
    alpha_et: f32,
    // Chain-local solar zenith angle [deg]: drives the VSPG importance and
    // the Dwivedi MIS ramps (port of the CPU sza_deg_local).
    sza_deg: f32,
    // Forced-collision gate: (sza >= ZENITH_SZA_START_DEG) && (!field_present
    // || field majorants packed). The 1D deck composes exactly via the
    // combined scout/advance channel; a 3D field composes via the
    // majorant-combined channel + truncated null-collision classification
    // (port of the CPU field_forced_classify; see twilight.metal for the
    // full derivation comments). Fields packed without majorants keep the
    // analog fallback.
    use_forced: u32,
    forced_tau_min: f32,
    cloud_channel: u32,
    beta_seed: f32,
    max_bounces: u32,
    g_seed: f32,
}

fn branch_params_for_sza(sza_deg: f32) -> BranchParams {
    let sza_t = clamp((sza_deg - ZENITH_SZA_START_DEG)
                      / (ZENITH_SZA_FULL_DEG - ZENITH_SZA_START_DEG), 0.0, 1.0);
    var bp: BranchParams;
    bp.zenith_frac = 0.5 + (ZENITH_MAX_FRACTION - 0.5) * sza_t;
    bp.n_zenith = 1.0 + (ZENITH_BIAS_N - 1.0) * sza_t;
    bp.term_share = TERMINATOR_MAX_SHARE * sza_t;
    bp.m_term = 1.0 + (TERMINATOR_N_MAX - 1.0) * sza_t;
    let tilt_deg = TERMINATOR_TILT_MIN_DEG
        + (TERMINATOR_TILT_MAX_DEG - TERMINATOR_TILT_MIN_DEG) * sza_t;
    bp.tilt_rad = tilt_deg * PI / 180.0;
    return bp;
}

// Terminator axis: unit vector tilted from up toward the sub-solar horizon.
fn terminator_axis(up: vec3<f32>, sun_dir: vec3<f32>, tilt_rad: f32) -> vec3<f32> {
    let dot_us = dot(sun_dir, up);
    let horiz = sun_dir - dot_us * up;
    let h_len = length(horiz);
    if (h_len < 1e-6) {
        return up;
    }
    let sun_horiz = horiz / h_len;
    let cos_t = cos(tilt_rad);
    let sin_t = sin(tilt_rad);
    return normalize(cos_t * up + sin_t * sun_horiz);
}

// ============================================================================
// Deep-twilight guiding helpers (port of the CPU photon.rs guiding stack;
// semantically twinned with the MSL helpers, see twilight.metal)
// ============================================================================

// Sigmoid for smooth SZA ramps (CPU sigmoid).
fn guide_sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// SZA-adaptive Dwivedi sampling fraction (CPU dwivedi_frac).
fn dwivedi_frac(sza_deg: f32) -> f32 {
    return DWIVEDI_FRAC_MAX * guide_sigmoid((sza_deg - DWIVEDI_SZA_CENTER) / DWIVEDI_SZA_WIDTH);
}

// SZA-adaptive Dwivedi concentration parameter (CPU dwivedi_beta).
fn dwivedi_beta(sza_deg: f32) -> f32 {
    return DWIVEDI_BETA_MAX * guide_sigmoid((sza_deg - DWIVEDI_SZA_CENTER) / DWIVEDI_SZA_WIDTH);
}

// Dwivedi PDF in sr^-1 for cos_z = dir . local_up (CPU dwivedi_pdf).
fn dwivedi_pdf(cos_z: f32, beta: f32) -> f32 {
    if (beta < 1e-6) {
        return INV_4PI;
    }
    let abs_cz = clamp(abs(cos_z), 0.0, 1.0);
    return beta * exp(-beta * abs_cz) / (4.0 * PI * (1.0 - exp(-beta)));
}

// Sample the Dwivedi distribution (CPU dwivedi_sample): CDF inversion of
// |cos_z|, random sign (symmetric about the horizontal plane), uniform phi.
struct DwivediSample {
    cos_z: f32,
    phi: f32,
}

fn dwivedi_sample(xi1: f32, xi2: f32, xi_sign: f32, beta: f32) -> DwivediSample {
    let phi = 2.0 * PI * xi2;
    if (beta < 1e-6) {
        return DwivediSample(2.0 * xi1 - 1.0, phi);
    }
    let one_minus_exp_neg_beta = 1.0 - exp(-beta);
    let abs_cz = clamp(-log(1.0 - xi1 * one_minus_exp_neg_beta) / beta, 0.0, 1.0);
    var cos_z = -abs_cz;
    if (xi_sign < 0.5) {
        cos_z = abs_cz;
    }
    return DwivediSample(cos_z, phi);
}

// Altitude/SZA-dependent VSPG importance (CPU vspg_importance): >= 1,
// ramping quadratically in altitude from 1.0 at 15 km to an SZA-dependent
// max (up to 50x) at 70 km.
fn vspg_importance(alt_m: f32, sza_deg: f32) -> f32 {
    if (alt_m <= VSPG_BOOST_START_M) {
        return 1.0;
    }
    let sza_t = clamp((sza_deg - VSPG_SZA_START) / (VSPG_SZA_FULL - VSPG_SZA_START), 0.0, 1.0);
    let alt_t = clamp((alt_m - VSPG_BOOST_START_M) / (VSPG_BOOST_FULL_M - VSPG_BOOST_START_M),
                      0.0, 1.0);
    let max_imp = 1.0 + (VSPG_MAX_IMPORTANCE - 1.0) * sza_t;
    return 1.0 + (max_imp - 1.0) * alt_t * alt_t;
}

// Per-shell VSPG segment (CPU VspgSegment).
struct VspgSegment {
    tau_lo: f32,
    tau_hi: f32,
    importance: f32,
}

// Per-shell cloud channel of the forced-flight scout/advance walk: the gray
// 1D deck (exact) without a field, the per-shell field MAJORANT with one
// (field runs carry all-zero atm cloud_extinction, so the two never mix).
fn chain_shell_cloud_ext(use_field_maj: bool, shell_idx: u32) -> f32 {
    if (use_field_maj) {
        return field_shell_majorant(shell_idx);
    }
    return read_cloud_extinction(shell_idx);
}

// ============================================================================
// Forced scattering scout, fused with VSPG segment collection: combined
// (gas + per-shell cloud channel) tau to boundary
// ============================================================================

struct ScoutResult {
    tau: f32,
    hit_ground: bool,
}

// Fused scout + VSPG segment collection (port of the CPU
// scout_with_vspg_segments; see the MSL twin for the overflow-rule
// derivation). Writes the per-shell segments and their count through the
// function-space pointers.
fn scout_with_vspg_segments(use_field_maj: bool, start_pos: vec3<f32>,
                            start_dir: vec3<f32>, wl_idx: u32, sza_deg: f32,
                            segments: ptr<function, array<VspgSegment, 64>>,
                            num_seg: ptr<function, u32>) -> ScoutResult {
    let ns = atm_num_shells();
    let surface_radius = atm[ATM_SHELLS_START];
    var pos = start_pos;
    var dir = start_dir;
    var tau = 0.0;
    *num_seg = 0u;

    let sidx = shell_index_binary(length(pos));
    if (sidx < 0) {
        return ScoutResult(0.0, false);
    }
    var us = u32(sidx);

    for (var iter = 0u; iter < 200u; iter++) {
        let shell_base = ATM_SHELLS_START + us * ATM_SHELL_STRIDE;
        let r_inner = atm[shell_base];
        let r_outer = atm[shell_base + 1u];
        let alt_mid = atm[shell_base + 2u];
        let extinction = atm[ATM_OPTICS_START + (us * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];

        let bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) {
            return ScoutResult(tau, false);
        }

        // COMBINED transport channel: gas + the per-shell cloud channel
        // (gray 1D shell deck, or the field MAJORANT in field-forced mode).
        let tau_shell = (extinction + chain_shell_cloud_ext(use_field_maj, us)) * bnd.dist;
        let tau_end = tau + tau_shell;

        // Collect the VSPG segment when the shell carries nonzero tau.
        if (*num_seg < VSPG_GPU_MAX_SEGMENTS && tau_shell > 1e-30) {
            (*segments)[*num_seg] = VspgSegment(tau, tau_end, vspg_importance(alt_mid, sza_deg));
            *num_seg = *num_seg + 1u;
        } else if (tau_shell > 1e-30) {
            // Segment-buffer OVERFLOW: extend the LAST segment across the
            // overflow tau at neutral importance so the set keeps tiling
            // [0, tau_max] (the CPU overflow rule; unbiased, coarser
            // importance on the tail).
            (*segments)[VSPG_GPU_MAX_SEGMENTS - 1u].tau_hi = tau_end;
            (*segments)[VSPG_GPU_MAX_SEGMENTS - 1u].importance = 1.0;
        }

        tau = tau_end;

        var boundary_pos = pos + dir * bnd.dist;
        let target_r = select(r_inner, r_outer, bnd.is_outward);
        let bp_r = length(boundary_pos);
        if (bp_r > 0.0) {
            boundary_pos *= (target_r / bp_r);
        }

        let inv_target_r = 1.0 / target_r;
        let n_from = read_refractive_index(us);
        let next_shell = select(us - 1u, us + 1u, bnd.is_outward);
        var n_to = 1.0;
        if (next_shell < ns) {
            n_to = read_refractive_index(next_shell);
        }
        dir = refract_at_boundary_r(dir, boundary_pos, inv_target_r, n_from, n_to);

        let radial = boundary_pos * inv_target_r;
        let nudge_sign = select(-1.0, 1.0, bnd.is_outward);
        pos = boundary_pos + radial * (nudge_sign * BOUNDARY_NUDGE_M);

        if (!bnd.is_outward && target_r <= surface_radius + 1.0) {
            return ScoutResult(tau, true);
        }
        if (next_shell >= ns) {
            return ScoutResult(tau, false);
        }
        us = next_shell;

        if (tau > FORCED_TAU_CUTOFF) {
            return ScoutResult(tau, false);
        }
    }

    return ScoutResult(tau, false);
}

// ============================================================================
// VSPG forced-flight sampler (port of the CPU vspg_sample_from_segments;
// see the MSL twin for the derivation comments). Returns (tau_s,
// weight_correction = I_avg / I_j); falls back to the plain truncated
// exponential (weight 1, ONE rng draw like the CPU fallback) when no
// segments exist or all probabilities vanish.
// ============================================================================

struct VspgSample {
    tau_s: f32,
    weight: f32,
}

fn vspg_sample_from_segments(segments: ptr<function, array<VspgSegment, 64>>,
                             num_seg: u32, tau_max: f32,
                             rng: ptr<function, vec2<u32>>) -> VspgSample {
    if (num_seg == 0u) {
        let xi = rng_next_f32(rng);
        let one_minus_exp = 1.0 - exp(-tau_max);
        return VspgSample(-log(1.0 - xi * one_minus_exp + 1e-30), 1.0);
    }

    var p_sum = 0.0;
    var q_sum = 0.0;
    for (var i = 0u; i < num_seg; i++) {
        let p_i = exp(-(*segments)[i].tau_lo) - exp(-(*segments)[i].tau_hi);
        p_sum += p_i;
        q_sum += (*segments)[i].importance * p_i;
    }

    if (q_sum < 1e-30) {
        let xi = rng_next_f32(rng);
        let one_minus_exp = 1.0 - exp(-tau_max);
        return VspgSample(-log(1.0 - xi * one_minus_exp + 1e-30), 1.0);
    }

    // CDF inversion: select segment j. Re-accumulates the SAME running sums
    // as the pass above (identical order, identical f32 values): the CPU's
    // q_cdf scan without the per-thread cdf array.
    let xi_segment = rng_next_f32(rng) * q_sum;
    var j = 0u;
    var q_run = (*segments)[0].importance
              * (exp(-(*segments)[0].tau_lo) - exp(-(*segments)[0].tau_hi));
    while (j + 1u < num_seg && q_run < xi_segment) {
        j++;
        q_run += (*segments)[j].importance
               * (exp(-(*segments)[j].tau_lo) - exp(-(*segments)[j].tau_hi));
    }

    // Within segment j: conditional truncated exponential, clamped for
    // numerical safety.
    let p_j = exp(-(*segments)[j].tau_lo) - exp(-(*segments)[j].tau_hi);
    let xi_within = rng_next_f32(rng);
    var tau_s = -log(exp(-(*segments)[j].tau_lo) - xi_within * p_j + 1e-30);
    tau_s = clamp(tau_s, (*segments)[j].tau_lo, (*segments)[j].tau_hi);

    // Weight correction I_avg / I_j with I_avg = q_sum / p_sum.
    let i_avg = q_sum / p_sum;
    return VspgSample(tau_s, i_avg / (*segments)[j].importance);
}

// ============================================================================
// Forced scattering: advance along a ray to a target COMBINED optical depth
// ============================================================================

struct AdvanceResult {
    pos: vec3<f32>,
    dir: vec3<f32>,
    shell_idx: u32,
}

fn advance_to_optical_depth(use_field_maj: bool, start_pos: vec3<f32>,
                            start_dir: vec3<f32>,
                            tau_target: f32, wl_idx: u32) -> AdvanceResult {
    let ns = atm_num_shells();
    let surface_radius = atm[ATM_SHELLS_START];
    var pos = start_pos;
    var dir = start_dir;
    var tau_acc = 0.0;

    let sidx = shell_index_binary(length(pos));
    if (sidx < 0) {
        return AdvanceResult(pos, dir, 0u);
    }
    var us = u32(sidx);

    for (var iter = 0u; iter < 200u; iter++) {
        let r_inner = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE];
        let r_outer = atm[ATM_SHELLS_START + us * ATM_SHELL_STRIDE + 1u];
        let optics_idx = us * MAX_WAVELENGTHS + wl_idx;
        // Combined per-shell extinction (gas + gray 1D cloud, or gas + the
        // per-shell field MAJORANT in field-forced mode), matching the
        // combined scout.
        let sigma_comb = atm[ATM_OPTICS_START + optics_idx * ATM_OPTICS_STRIDE]
            + chain_shell_cloud_ext(use_field_maj, us);

        let bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) {
            return AdvanceResult(pos, dir, us);
        }

        let tau_shell = sigma_comb * bnd.dist;

        if (tau_acc + tau_shell >= tau_target) {
            // Scatter point is within this shell.
            let tau_remaining = tau_target - tau_acc;
            var dist = bnd.dist;
            if (sigma_comb > 1e-30) {
                dist = tau_remaining / sigma_comb;
            }
            pos = pos + dir * dist;
            return AdvanceResult(pos, dir, us);
        }

        // Cross boundary.
        tau_acc += tau_shell;
        var boundary_pos = pos + dir * bnd.dist;
        boundary_pos = snap_to_radius(boundary_pos, select(r_inner, r_outer, bnd.is_outward));
        let n_from = read_refractive_index(us);
        let next_shell = select(us - 1u, us + 1u, bnd.is_outward);
        var n_to = 1.0;
        if (next_shell < ns) {
            n_to = read_refractive_index(next_shell);
        }
        dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
        pos = radial_nudge(boundary_pos, bnd.is_outward);

        if (!bnd.is_outward && length(pos) <= surface_radius + 1.0) {
            return AdvanceResult(pos, dir, us);
        }
        if (next_shell >= ns) {
            return AdvanceResult(pos, dir, us);
        }
        us = next_shell;
    }

    return AdvanceResult(pos, dir, us);
}

// ============================================================================
// Secondary chain tracer (used by the hybrid_scatter_v2 kernel)
//
// Full Stokes [I,Q,U,V] propagation; unbiased one-sample-MIS seed; gray
// cloud channel (field DDA or analytic 1D inversion) racing the gas
// channel; combined-channel forced mode for the 1D deck. Faithful port of
// the MSL trace_secondary_chain (see that function and the CPU
// trace_secondary_chain for the estimator derivations).
// ============================================================================

fn trace_secondary_chain(field_present: bool, start_pos: vec3<f32>,
                         sun_dir: vec3<f32>, wl_idx: u32,
                         start_optics: ShellOptics, prev_dir_in: vec3<f32>,
                         setup: SecondarySetup,
                         ray_idx: u32, total_rays: u32,
                         rng: ptr<function, vec2<u32>>) -> vec4<f32> {
    let surface_radius = atm_surface_radius();

    // Unbiased one-sample-MIS seed: sample omega from the 3-component
    // mixture, weight by the balance-heuristic estimator.
    let xi_jitter = rng_next_f32(rng);
    let xi_mix = (f32(ray_idx) + xi_jitter) / max(f32(total_rays), 1.0);
    var dir: vec3<f32>;
    if (xi_mix < setup.alpha_p) {
        var ct: f32;
        if (rng_next_f32(rng) < start_optics.rayleigh_fraction) {
            ct = sample_rayleigh_analytic(rng_next_f32(rng));
        } else {
            ct = sample_henyey_greenstein(rng_next_f32(rng), start_optics.asymmetry);
        }
        let phi_init = 2.0 * PI * rng_next_f32(rng);
        dir = scatter_direction(sun_dir, ct, phi_init);
    } else if (xi_mix < setup.alpha_p + setup.alpha_z || setup.alpha_t < 1e-6) {
        let zs = sample_zenith_biased(setup.local_up, setup.n_zenith, rng);
        dir = zs.dir;
    } else {
        let zs = sample_zenith_biased(setup.term_axis_dir, setup.m_term, rng);
        dir = zs.dir;
    }

    let q_seed = seed_mixture_pdf(dir, sun_dir, setup.local_up,
                                  setup.term_axis_dir, setup.alpha_p,
                                  setup.alpha_z, setup.alpha_t,
                                  setup.n_zenith, setup.m_term, start_optics);
    // Cloud-seed mixture: one type draw selects the vertex type with
    // p_c = beta_cloud / beta_total; consumed ONLY when the seeding substep
    // carries cloud (clear-sky RNG streams keep their structure).
    var seed_is_cloud = false;
    if (setup.beta_seed > 0.0) {
        let beta_gas_seed = start_optics.extinction * start_optics.ssa;
        let p_c = setup.beta_seed / (setup.beta_seed + beta_gas_seed);
        seed_is_cloud = rng_next_f32(rng) < p_c;
    }
    // prev_dir_in is the LOS view direction; a cloud seed swaps the
    // numerator phase to the gray HG lobe against the SAME q.
    var phase_seed: f32;
    if (seed_is_cloud) {
        phase_seed = henyey_greenstein_phase(dot(dir, prev_dir_in), setup.g_seed);
    } else {
        phase_seed = mixed_phase(dot(dir, prev_dir_in), start_optics);
    }
    var w0 = 0.0;
    if (q_seed > 1e-30) {
        w0 = phase_seed * INV_4PI / q_seed;
    }

    // Seed polarization: unpolarized (same approximation as the CPU chain).
    var stokes = vec4<f32>(1.0, 0.0, 0.0, 0.0);

    var pos = start_pos;
    var current_dir = dir;
    var prev_dir = sun_dir; // direction before current propagation segment
    var weight = w0;

    var total_I = kahan_new();
    var total_Q = kahan_new();
    var total_U = kahan_new();
    var total_V = kahan_new();

    // Deep-twilight guiding parameters (port of the CPU chains; see the
    // MSL twin). The field-forced (majorant-combined) channel is active
    // exactly when the kernel enabled forced mode under a field.
    let d_frac = dwivedi_frac(setup.sza_deg);
    let d_beta = dwivedi_beta(setup.sza_deg);
    let use_field_maj = field_present && (setup.use_forced != 0u);

    for (var scatter_iter = 0u; scatter_iter < setup.max_bounces; scatter_iter++) {
        // --- Decide scatter mode for this bounce ---
        // Fused scout + VSPG: one shell walk collects tau_max AND the
        // altitude/SZA-importance segments for the forced-flight sampler.
        var forced_this_bounce = false;
        var tau_max = 0.0;
        var vspg_segs: array<VspgSegment, 64>;
        var n_vspg_segs = 0u;

        if (setup.use_forced != 0u) {
            let scout = scout_with_vspg_segments(
                use_field_maj, pos, current_dir, wl_idx, setup.sza_deg,
                &vspg_segs, &n_vspg_segs);
            tau_max = scout.tau;
            // Force scatter only when the path exits to space and tau is in
            // the useful range (see twilight.metal for the rationale).
            forced_this_bounce = !scout.hit_ground
                && tau_max >= setup.forced_tau_min
                && tau_max < FORCED_TAU_CUTOFF;
        }

        var scatter_shell = 0u;
        // Gray cloud channel: a cloud collision is a distinct vertex type
        // (pure depolarizing HG scatter, no SSA, no weight change).
        var cloud_collision = false;
        var g_cloud_here = 0.0;

        if (forced_this_bounce) {
            // Upfront forced scattering (unbiased); tau_max is the
            // (majorant-)COMBINED optical depth (gas + gray 1D deck, or
            // gas + per-shell field majorant).
            let exp_neg_tau = exp(-tau_max);
            weight *= (1.0 - exp_neg_tau);
            if (weight < 1e-30) {
                break;
            }
            // VSPG: sample the collision location from the pre-collected
            // importance segments (weight-corrected, unbiased; replaces
            // the plain truncated-exponential draw).
            let vs = vspg_sample_from_segments(&vspg_segs, n_vspg_segs, tau_max, rng);
            let tau_s = vs.tau_s;
            weight *= vs.weight;
            let adv = advance_to_optical_depth(use_field_maj, pos, current_dir, tau_s, wl_idx);
            pos = adv.pos;
            current_dir = adv.dir;
            scatter_shell = adv.shell_idx;

            if (use_field_maj) {
                // FIELD: classify the majorant collision (real cloud / real
                // gas / null); nulls re-draw within the remaining truncated
                // budget. Port of the CPU field_forced_classify (see
                // twilight.metal for the derivation and the f32 kill
                // threshold note). The classification uniform is drawn ONLY
                // in shells with a positive cloud majorant.
                var consumed = tau_s;
                var fshell = adv.shell_idx;
                var resolved = false;
                for (var ev = 0u; ev < FIELD_NULL_EVENT_LIMIT; ev++) {
                    let c_maj = field_shell_majorant(fshell);
                    if (c_maj <= 0.0) { // real gas with probability 1
                        resolved = true;
                        break;
                    }
                    let sigma_gas_m = atm[ATM_OPTICS_START
                        + (fshell * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];
                    let sigma_m = sigma_gas_m + c_maj;
                    let sigma_c_here = field_sigma_at(pos);
                    let xi_cls = rng_next_f32(rng) * sigma_m;
                    if (xi_cls < sigma_c_here) {
                        cloud_collision = true;
                        g_cloud_here = field_g_at(pos);
                        resolved = true;
                        break;
                    }
                    if (xi_cls < sigma_c_here + sigma_gas_m) {
                        resolved = true;
                        break;
                    }
                    // NULL: continue the truncated flight in the remaining
                    // budget; kill on an fp-exhausted budget (f32 threshold,
                    // see the MSL twin).
                    let t_rem = tau_max - consumed;
                    if (t_rem <= 1e-6) {
                        break; // killed below
                    }
                    let e_rem = exp(-t_rem);
                    weight *= (1.0 - e_rem);
                    let xi2 = rng_next_f32(rng);
                    let d_tau = -log(1.0 - xi2 * (1.0 - e_rem) + 1e-30);
                    let nadv = advance_to_optical_depth(
                        use_field_maj, pos, current_dir, d_tau, wl_idx);
                    pos = nadv.pos;
                    current_dir = nadv.dir;
                    fshell = nadv.shell_idx;
                    consumed += d_tau;
                }
                scatter_shell = fshell;
                if (!resolved) {
                    // Backstop kill (fp-exhausted budget or the null-event
                    // limit): terminate the particle with weight zero.
                    weight = 0.0;
                    break;
                }
            } else {
                // 1D deck: vertex type from the exact extinction conditional
                // at the collision shell; draw taken ONLY when the shell
                // carries cloud.
                let sigma_c_f = read_cloud_extinction(scatter_shell);
                if (sigma_c_f > 0.0) {
                    let sigma_gas_f = atm[ATM_OPTICS_START
                        + (scatter_shell * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];
                    if (rng_next_f32(rng) < sigma_c_f / (sigma_c_f + sigma_gas_f)) {
                        cloud_collision = true;
                        // The gray deck's delta-scaled asymmetry.
                        g_cloud_here = atm[ATM_CLOUD_G_SCALED];
                    }
                }
            }
        } else {
            // Analog scatter WITH the gray cloud channel (decomposition
            // tracking): the gas channel keeps the exponential transform, a
            // separate gray cloud Poisson process races over each segment.
            // Cloud budget drawn ONCE per free flight from the SAME stream.
            var scatter_found = false;
            var found_shell = 0u;
            var tau_c_remaining = -log(1.0 - rng_next_f32(rng) + 1e-30);

            for (var walk_i = 0u; walk_i < 200u; walk_i++) {
                let r = length(pos);
                let sidx = shell_index_binary(r);
                if (sidx < 0) {
                    break;
                }

                let us = u32(sidx);
                let sh = read_shell(us);
                let op = read_optics(us, wl_idx);

                let bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
                if (!bnd.found) {
                    break;
                }

                // Gas free path (exponential transform on the gas channel);
                // clear air walks to the boundary (no gas tau draw).
                let sigma = op.extinction;
                var cos_bias = 0.0;
                var sigma_prime = sigma;
                var free_path = FREE_PATH_INF;
                if (sigma >= 1e-20) {
                    cos_bias = dot(current_dir, setup.term_axis_dir);
                    sigma_prime = sigma * (1.0 - setup.alpha_et * cos_bias);
                    if (sigma_prime <= 0.0) {
                        sigma_prime = sigma;
                    }
                    let xi = rng_next_f32(rng);
                    free_path = -log(1.0 - xi + 1e-30) / sigma_prime;
                }

                // Race the gray cloud channel over the segment up to the gas
                // event (gas scatter at free_path or boundary crossing).
                let gas_cap = min(free_path, bnd.dist);
                var cloud_dist = -1.0;
                var tau_pass = 0.0;
                if (field_present) {
                    cloud_dist = field_advance_to_tau(pos, current_dir, gas_cap, tau_c_remaining);
                } else {
                    let sigma_c = read_cloud_extinction(us);
                    if (sigma_c > 0.0) {
                        let dist_c = tau_c_remaining / sigma_c;
                        if (dist_c <= gas_cap) {
                            cloud_dist = dist_c;
                        } else {
                            tau_pass = sigma_c * gas_cap;
                        }
                    }
                }
                if (cloud_dist >= 0.0) {
                    // Cloud wins. ET gas weight correction for the distance
                    // actually travelled (gray cloud ratio = 1).
                    if (setup.alpha_et > 0.0 && sigma >= 1e-20) {
                        let et_arg = -setup.alpha_et * sigma * cos_bias * cloud_dist;
                        if (abs(et_arg) < 80.0) {
                            weight *= exp(et_arg);
                        } else {
                            weight = 0.0;
                        }
                    }
                    if (!is_finite_f32(weight)) {
                        break;
                    }
                    pos = pos + current_dir * cloud_dist;
                    if (field_present) {
                        g_cloud_here = field_g_at(pos);
                    } else {
                        g_cloud_here = atm[ATM_CLOUD_G_SCALED];
                    }
                    found_shell = us;
                    scatter_found = true;
                    cloud_collision = true;
                    break;
                } else {
                    // No cloud collision in this segment: consume its tau.
                    if (field_present) {
                        tau_pass = field_tau_along(pos, current_dir, gas_cap);
                    }
                    tau_c_remaining -= tau_pass;
                }

                if (free_path >= bnd.dist) {
                    if (setup.alpha_et > 0.0 && sigma >= 1e-20) {
                        let et_arg = -setup.alpha_et * sigma * cos_bias * bnd.dist;
                        if (abs(et_arg) < 80.0) {
                            weight *= exp(et_arg);
                        } else {
                            weight = 0.0;
                        }
                    }
                    if (!is_finite_f32(weight)) {
                        break;
                    }

                    var boundary_pos = pos + current_dir * bnd.dist;
                    boundary_pos = snap_to_radius(
                        boundary_pos, select(sh.r_inner, sh.r_outer, bnd.is_outward));

                    // Ground reflection.
                    if (!bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                        let normal = normalize(boundary_pos);
                        // Snap the bounce point ABOVE the surface (see the
                        // MSL comment: the CPU 1 mm ledge vanishes in f32).
                        let ground_pos = normal * (surface_radius + BOUNDARY_NUDGE_M);
                        let cos_sun_ground = dot(sun_dir, normal);
                        if (cos_sun_ground > 0.0) {
                            let t_sun_gb = shadow_ray_transmittance_chain(
                                field_present, setup.cloud_channel != 0u,
                                ground_pos, sun_dir, wl_idx);
                            if (t_sun_gb > 1e-30) {
                                let albedo_nee = read_albedo(wl_idx);
                                let nee_gb = weight * albedo_nee * t_sun_gb * cos_sun_ground / PI;
                                if (is_finite_f32(nee_gb)) {
                                    kahan_add(&total_I, nee_gb);
                                }
                            }
                        }
                        let albedo = read_albedo(wl_idx);
                        weight *= albedo;
                        if (!is_finite_f32(weight) || abs(weight) < 1e-30) {
                            break;
                        }
                        prev_dir = current_dir;
                        current_dir = sample_hemisphere(normal, rng);
                        pos = ground_pos;
                        stokes = vec4<f32>(1.0, 0.0, 0.0, 0.0);
                        // New free flight: redraw the cloud budget.
                        tau_c_remaining = -log(1.0 - rng_next_f32(rng) + 1e-30);
                        continue;
                    }

                    // Refract and cross into the next shell; the cloud
                    // budget carries over undiminished.
                    let n_from = read_refractive_index(us);
                    let next_s = select(us - 1u, us + 1u, bnd.is_outward);
                    var n_to = 1.0;
                    if (next_s < atm_num_shells()) {
                        n_to = read_refractive_index(next_s);
                    }
                    current_dir = refract_at_boundary(current_dir, boundary_pos, n_from, n_to);
                    pos = radial_nudge(boundary_pos, bnd.is_outward);
                    continue;
                }

                // Gas scatter within this shell.
                if (setup.alpha_et > 0.0 && sigma >= 1e-20) {
                    let et_arg = -setup.alpha_et * sigma * cos_bias * free_path;
                    if (abs(et_arg) < 80.0) {
                        weight *= (sigma / sigma_prime) * exp(et_arg);
                    } else {
                        weight = 0.0;
                    }
                }
                if (!is_finite_f32(weight)) {
                    break;
                }
                pos = pos + current_dir * free_path;
                found_shell = us;
                scatter_found = true;
                break;
            }

            if (!scatter_found) {
                break;
            }
            scatter_shell = found_shell;
        }

        let op = read_optics(scatter_shell, wl_idx);

        // SSA: a cloud collision is pure scattering (no SSA factor).
        if (!cloud_collision) {
            weight *= op.ssa;
        }

        // NEE. A cloud vertex is a depolarizing HG (phase on I, output
        // unpolarized); a gas vertex applies the Mueller matrix to the
        // photon's actual Stokes state.
        if (is_finite_f32(weight) && abs(weight) > 1e-30) {
            let t_sun_sec = shadow_ray_transmittance_chain(
                field_present, setup.cloud_channel != 0u, pos, sun_dir, wl_idx);
            if (t_sun_sec > 1e-30) {
                let cos_angle_nee = clamp(dot(sun_dir, current_dir), -1.0, 1.0);
                var nee_stokes: vec4<f32>;
                if (cloud_collision) {
                    let p = henyey_greenstein_phase(cos_angle_nee, g_cloud_here);
                    nee_stokes = vec4<f32>(stokes.x * p, 0.0, 0.0, 0.0);
                } else {
                    var A_nee = 0.0;
                    var B_nee = 0.0;
                    var C_nee = 0.0;
                    stokes_ABC(cos_angle_nee, op, &A_nee, &B_nee, &C_nee);
                    var cos2phi_nee = 0.0;
                    var sin2phi_nee = 0.0;
                    scattering_plane_rotation(prev_dir, current_dir, -sun_dir,
                                              &cos2phi_nee, &sin2phi_nee);
                    if (!is_finite_f32(cos2phi_nee)) {
                        cos2phi_nee = 1.0;
                        sin2phi_nee = 0.0;
                    }
                    nee_stokes = scatter_stokes(A_nee, B_nee, C_nee,
                                                cos2phi_nee, sin2phi_nee, stokes);
                }

                let scale = weight * t_sun_sec / (4.0 * PI);
                if (is_finite_f32(scale)) {
                    let nee_I = scale * nee_stokes.x;
                    if (is_finite_f32(nee_I)) { kahan_add(&total_I, nee_I); }
                    let nee_Q = scale * nee_stokes.y;
                    if (is_finite_f32(nee_Q)) { kahan_add(&total_Q, nee_Q); }
                    let nee_U = scale * nee_stokes.z;
                    if (is_finite_f32(nee_U)) { kahan_add(&total_U, nee_U); }
                    let nee_V = scale * nee_stokes.w;
                    if (is_finite_f32(nee_V)) { kahan_add(&total_V, nee_V); }
                }
            }
        }

        if (!is_finite_f32(weight) || abs(weight) < 1e-30) {
            break;
        }

        // Sample the new direction. A cloud vertex scatters from the gray
        // HG lobe and resets polarization; a gas vertex samples the
        // Rayleigh/HG mixture and updates Stokes.
        if (cloud_collision) {
            let ct_cloud = sample_henyey_greenstein(rng_next_f32(rng), g_cloud_here);
            let phi_cloud = 2.0 * PI * rng_next_f32(rng);
            let d_cloud = scatter_direction(current_dir, ct_cloud, phi_cloud);
            if (!is_finite_f32(d_cloud.x) || (length(d_cloud) < 1e-10)) {
                break;
            }
            prev_dir = current_dir;
            current_dir = d_cloud;
            stokes = vec4<f32>(1.0, 0.0, 0.0, 0.0);
            continue;
        }

        // Sample new direction (gas vertex): Dwivedi/phase MIS mixture at
        // deep twilight (port of the CPU chains' MIS block; the balance-
        // heuristic weight corrects the SCALAR intensity, the Stokes update
        // below uses the actual sampled angle). Below the 0.02 activation:
        // the pure-phase path, draw-for-draw identical to history.
        var cos_theta: f32;
        var new_dir: vec3<f32>;
        let mis_active = d_frac >= 0.02;
        if (mis_active) {
            let local_up_here = normalize(pos);
            let alpha_p_mis = 1.0 - d_frac;
            let xi_branch = rng_next_f32(rng);
            if (xi_branch < d_frac) {
                // Dwivedi branch: horizontal-biased escape sampling in the
                // local (up, east, north) frame.
                let xi1 = rng_next_f32(rng);
                let xi2 = rng_next_f32(rng);
                let xi_sign = rng_next_f32(rng);
                let dw = dwivedi_sample(xi1, xi2, xi_sign, d_beta);
                let sin_z = sqrt(max(1.0 - dw.cos_z * dw.cos_z, 0.0));
                var arbitrary = vec3<f32>(1.0, 0.0, 0.0);
                if (abs(local_up_here.y) < 0.9) {
                    arbitrary = vec3<f32>(0.0, 1.0, 0.0);
                }
                let east = normalize(cross(local_up_here, arbitrary));
                let north = cross(local_up_here, east);
                let d = normalize(local_up_here * dw.cos_z
                                  + east * (sin_z * cos(dw.phi))
                                  + north * (sin_z * sin(dw.phi)));
                cos_theta = clamp(dot(current_dir, d), -1.0, 1.0);
                let p_phase = mixed_phase(cos_theta, op) * INV_4PI;
                let p_dw = dwivedi_pdf(dw.cos_z, d_beta);
                let mis_denom = alpha_p_mis * p_phase + d_frac * p_dw;
                if (mis_denom > 1e-30) {
                    weight *= p_phase / mis_denom;
                }
                new_dir = d;
            } else {
                // Phase branch (within MIS).
                if (rng_next_f32(rng) < op.rayleigh_fraction) {
                    cos_theta = sample_rayleigh_analytic(rng_next_f32(rng));
                } else {
                    cos_theta = sample_henyey_greenstein(rng_next_f32(rng), op.asymmetry);
                }
                cos_theta = clamp(cos_theta, -1.0, 1.0);
                let phi = 2.0 * PI * rng_next_f32(rng);
                new_dir = scatter_direction(current_dir, cos_theta, phi);
                let p_phase = mixed_phase(cos_theta, op) * INV_4PI;
                let cos_z_dw = dot(new_dir, local_up_here);
                let p_dw = dwivedi_pdf(cos_z_dw, d_beta);
                let mis_denom = alpha_p_mis * p_phase + d_frac * p_dw;
                if (mis_denom > 1e-30) {
                    weight *= p_phase / mis_denom;
                }
            }
        } else {
            // Pure phase function: no Dwivedi, no MIS overhead.
            if (rng_next_f32(rng) < op.rayleigh_fraction) {
                cos_theta = sample_rayleigh_analytic(rng_next_f32(rng));
            } else {
                cos_theta = sample_henyey_greenstein(rng_next_f32(rng), op.asymmetry);
            }
            cos_theta = clamp(cos_theta, -1.0, 1.0);
            let phi = 2.0 * PI * rng_next_f32(rng);
            new_dir = scatter_direction(current_dir, cos_theta, phi);
        }
        if (!is_finite_f32(new_dir.x) || (length(new_dir) < 1e-10)) {
            break;
        }

        // Update Stokes state through this scatter event.
        var A_s = 0.0;
        var B_s = 0.0;
        var C_s = 0.0;
        stokes_ABC(cos_theta, op, &A_s, &B_s, &C_s);
        var cos2phi_s = 0.0;
        var sin2phi_s = 0.0;
        scattering_plane_rotation(prev_dir, current_dir, new_dir, &cos2phi_s, &sin2phi_s);
        if (!is_finite_f32(cos2phi_s)) {
            cos2phi_s = 1.0;
            sin2phi_s = 0.0;
        }
        stokes = scatter_stokes(A_s, B_s, C_s, cos2phi_s, sin2phi_s, stokes);

        // Normalize by I (importance weighting -- keeps stokes.x = 1).
        if (is_finite_f32(stokes.x) && stokes.x > 1e-30) {
            let inv_I = 1.0 / stokes.x;
            stokes *= inv_I;
            if (!is_finite_f32(stokes.y)) { stokes.y = 0.0; }
            if (!is_finite_f32(stokes.z)) { stokes.z = 0.0; }
            if (!is_finite_f32(stokes.w)) { stokes.w = 0.0; }
        } else {
            stokes = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        }

        prev_dir = current_dir;
        current_dir = new_dir;
    }

    var result = vec4<f32>(kahan_result(total_I), kahan_result(total_Q),
                           kahan_result(total_U), kahan_result(total_V));
    // Only filter non-finite values (numerical safety, not bias).
    if (!is_finite_f32(result.x)) { result.x = 0.0; }
    if (!is_finite_f32(result.y)) { result.y = 0.0; }
    if (!is_finite_f32(result.z)) { result.z = 0.0; }
    if (!is_finite_f32(result.w)) { result.w = 0.0; }
    return result;
}

// ============================================================================
// Kernel 1: single_scatter_spectrum
//
// One thread per wavelength. Full LOS integration with refracted shadow
// rays. Kahan summation for optical depth and radiance accumulation.
// ============================================================================

@compute @workgroup_size(256)
fn single_scatter_spectrum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (!atm_header_valid()) {
        if (tid == 0u) {
            out_buf[0] = HEADER_SENTINEL;
        }
        return;
    }
    let num_wl = atm_num_wavelengths();
    if (tid >= num_wl) {
        return;
    }

    let wl_idx = tid;
    let observer_pos = read_observer();
    let view_dir = read_view_dir();
    let sun_dir = read_sun_dir();

    let toa_radius = atm_toa_radius();
    let surface_radius = atm_surface_radius();

    // Find LOS extent.
    let toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
    if (!toa_hit.hit || toa_hit.t_far <= 0.0) {
        out_buf[tid] = 0.0;
        return;
    }
    let los_max = toa_hit.t_far;

    let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    let hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3 && ground_hit.t_near < los_max;
    let los_end = select(los_max, ground_hit.t_near, hits_ground);

    if (los_end <= 0.0) {
        out_buf[tid] = 0.0;
        return;
    }

    let num_steps = min(MAX_LOS_STEPS, u32(los_end / 500.0) + 20u);
    let ds = los_end / f32(num_steps);

    var radiance = kahan_new();
    var tau_obs = kahan_new();
    // Cloud portion of the eye path: Eddington diffuse (broadband).
    var tau_cloud_obs = 0.0;

    let cos_theta = dot(sun_dir, view_dir);

    for (var step_i = 0u; step_i < num_steps; step_i++) {
        let s = (f32(step_i) + 0.5) * ds;
        let scatter_pos = observer_pos + view_dir * s;
        let r = length(scatter_pos);

        if (r > toa_radius || r < surface_radius) {
            continue;
        }

        let sidx = shell_index_binary(r);
        if (sidx < 0) {
            continue;
        }

        let op = read_optics(u32(sidx), wl_idx);
        let cloud_ext_step = read_cloud_extinction(u32(sidx));
        let beta_scat = op.extinction * op.ssa;

        if (beta_scat < 1e-30) {
            kahan_add(&tau_obs, op.extinction * ds);
            tau_cloud_obs += cloud_ext_step * ds;
            continue;
        }

        // Single exp(-(tau + half_step)) (see the MSL precision comment).
        let tau_cloud_mid = tau_cloud_obs + cloud_ext_step * ds * 0.5;
        let t_obs = exp(-(kahan_result(tau_obs) + op.extinction * ds * 0.5))
            * cloud_diffuse_transmittance(tau_cloud_mid);

        if (t_obs < 1e-30) {
            break;
        }

        let t_sun = shadow_ray_transmittance(scatter_pos, sun_dir, wl_idx);

        if (t_sun < 1e-30) {
            kahan_add(&tau_obs, op.extinction * ds);
            tau_cloud_obs += cloud_ext_step * ds;
            continue;
        }

        let phase = mixed_phase(cos_theta, op);
        let di = beta_scat * phase / (4.0 * PI) * t_sun * t_obs * ds;
        kahan_add(&radiance, di);

        kahan_add(&tau_obs, op.extinction * ds);
        tau_cloud_obs += cloud_ext_step * ds;
    }

    // Ground reflection (Lambertian BRDF = albedo / pi).
    if (hits_ground) {
        let albedo = read_albedo(wl_idx);
        if (albedo > 1e-10) {
            let ground_pos = observer_pos + view_dir * los_end;
            let ground_normal = normalize(ground_pos);
            let cos_sun_incidence = dot(sun_dir, ground_normal);

            if (cos_sun_incidence > 0.0) {
                let t_sun_ground = shadow_ray_transmittance(ground_pos, sun_dir, wl_idx);
                let t_obs_ground = exp(-kahan_result(tau_obs))
                    * cloud_diffuse_transmittance(tau_cloud_obs);
                kahan_add(&radiance, albedo / PI * cos_sun_incidence * t_sun_ground * t_obs_ground);
            }
        }
    }

    out_buf[tid] = kahan_result(radiance);
}

// ============================================================================
// Kernel 2: mcrt_trace_photon
//
// One thread per (wavelength, photon) pair; output per-thread weight, the
// host reduces. The Multiple estimator has no cloud channel on ANY GPU
// backend (the pipeline routes cloudy Multiple runs to CPU), matching
// Metal.
// ============================================================================

@compute @workgroup_size(256)
fn mcrt_trace_photon(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (!atm_header_valid()) {
        if (tid == 0u) {
            out_buf[0] = HEADER_SENTINEL;
        }
        return;
    }
    let num_wl = atm_num_wavelengths();
    let photons_per_wl = read_photons_per_wl();
    let total_threads = num_wl * photons_per_wl;
    if (tid >= total_threads) {
        return;
    }

    let wl_idx = tid / photons_per_wl;
    let photon_idx = tid % photons_per_wl;

    let observer_pos = read_observer();
    let view_dir = read_view_dir();
    let sun_dir = read_sun_dir();

    // Unique seed per (wavelength, photon) pair: the exact Metal LCG-style
    // derivation on the emulated u64.
    let base_seed = read_rng_seed();
    var rng = u64_add(base_seed, vec2<u32>(wl_idx, 0u));
    rng = u64_mul(rng, LCG_K1);
    rng = u64_add(rng, vec2<u32>(photon_idx, 0u));
    rng = u64_mul(rng, LCG_K2);
    rng = u64_add(rng, vec2<u32>(1u, 0u));

    let surface_radius = atm_surface_radius();

    var pos = observer_pos;
    // Surface snap (port of the CPU trace_photon entry snap; see MSL).
    {
        let r0 = length(pos);
        let ledge = surface_radius + BOUNDARY_NUDGE_M;
        if (r0 > 0.0 && r0 < ledge) {
            pos = pos * (ledge / r0);
        }
    }
    var dir = view_dir;
    var prev_dir = dir; // for Stokes scattering plane tracking
    var stokes = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    var weight = 1.0;
    var result_weight = kahan_new();

    for (var bounce = 0u; bounce < MAX_SCATTERS; bounce++) {
        let r = length(pos);
        let sidx = shell_index_binary(r);
        if (sidx < 0) {
            break;
        }

        let us = u32(sidx);
        let sh = read_shell(us);
        let op = read_optics(us, wl_idx);

        if (op.extinction < 1e-20) {
            let bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
            if (!bnd.found) {
                break;
            }
            var boundary_pos = pos + dir * bnd.dist;
            boundary_pos = snap_to_radius(
                boundary_pos, select(sh.r_inner, sh.r_outer, bnd.is_outward));
            let n_from = read_refractive_index(us);
            let next_s = select(us - 1u, us + 1u, bnd.is_outward);
            var n_to = 1.0;
            if (next_s < atm_num_shells()) {
                n_to = read_refractive_index(next_s);
            }
            dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        // Sample free path: -log(xi) with xi in (0, 1].
        let xi = rng_next_f32(&rng);
        let free_path = -log(xi) / op.extinction;

        let bnd = next_shell_boundary(pos, dir, sh.r_inner, sh.r_outer);
        if (!bnd.found) {
            break;
        }

        if (free_path >= bnd.dist) {
            // Exit shell without scattering.
            var boundary_pos = pos + dir * bnd.dist;
            boundary_pos = snap_to_radius(
                boundary_pos, select(sh.r_inner, sh.r_outer, bnd.is_outward));

            // Ground reflection: depolarizes.
            if (!bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                let albedo = read_albedo(wl_idx);
                let normal = normalize(boundary_pos);
                // Ground-bounce NEE (Lambertian albedo/pi), BEFORE the
                // albedo folds into the continuing weight.
                let cos_sun_g = dot(sun_dir, normal);
                if (cos_sun_g > 0.0) {
                    let t_sun_g = shadow_ray_transmittance(
                        boundary_pos + normal * BOUNDARY_NUDGE_M, sun_dir, wl_idx);
                    if (t_sun_g > 1e-30) {
                        kahan_add(&result_weight,
                                  weight * albedo * t_sun_g * cos_sun_g * (1.0 / PI));
                    }
                }
                weight *= albedo;
                prev_dir = dir;
                dir = sample_hemisphere(normal, &rng);
                pos = radial_nudge(boundary_pos, true);
                stokes = vec4<f32>(1.0, 0.0, 0.0, 0.0);
                continue;
            }

            // Refract and nudge past boundary.
            {
                let n_from = read_refractive_index(us);
                let next_s = select(us - 1u, us + 1u, bnd.is_outward);
                var n_to = 1.0;
                if (next_s < atm_num_shells()) {
                    n_to = read_refractive_index(next_s);
                }
                dir = refract_at_boundary(dir, boundary_pos, n_from, n_to);
            }
            pos = radial_nudge(boundary_pos, bnd.is_outward);
            continue;
        }

        // Scattering event.
        pos = pos + dir * free_path;

        // SSA BEFORE NEE (CPU convention; see MSL comment).
        weight *= op.ssa;

        // NEE: apply Mueller to the photon's current Stokes state.
        let t_sun = shadow_ray_transmittance(pos, sun_dir, wl_idx);
        if (t_sun > 1e-30) {
            let cos_angle = dot(sun_dir, dir);
            var A_nee = 0.0;
            var B_nee = 0.0;
            var C_nee = 0.0;
            stokes_ABC(cos_angle, op, &A_nee, &B_nee, &C_nee);
            var cos2phi_nee = 0.0;
            var sin2phi_nee = 0.0;
            scattering_plane_rotation(prev_dir, dir, -sun_dir, &cos2phi_nee, &sin2phi_nee);
            let nee_stokes = scatter_stokes(A_nee, B_nee, C_nee,
                                            cos2phi_nee, sin2phi_nee, stokes);
            kahan_add(&result_weight, weight * t_sun * nee_stokes.x / (4.0 * PI));
        }

        // Sample new direction and update Stokes state.
        var cos_theta: f32;
        if (rng_next_f32(&rng) < op.rayleigh_fraction) {
            cos_theta = sample_rayleigh_analytic(rng_next_f32(&rng));
        } else {
            cos_theta = sample_henyey_greenstein(rng_next_f32(&rng), op.asymmetry);
        }
        let phi = 2.0 * PI * rng_next_f32(&rng);
        let new_dir = scatter_direction(dir, cos_theta, phi);

        var A_s = 0.0;
        var B_s = 0.0;
        var C_s = 0.0;
        stokes_ABC(cos_theta, op, &A_s, &B_s, &C_s);
        var cos2phi_s = 0.0;
        var sin2phi_s = 0.0;
        scattering_plane_rotation(prev_dir, dir, new_dir, &cos2phi_s, &sin2phi_s);
        stokes = scatter_stokes(A_s, B_s, C_s, cos2phi_s, sin2phi_s, stokes);
        if (stokes.x > 1e-30) {
            stokes *= 1.0 / stokes.x;
        } else {
            stokes = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        }

        prev_dir = dir;
        dir = new_dir;
    }

    out_buf[tid] = kahan_result(result_weight);
}

// ============================================================================
// Kernel 3a: hybrid_context_prefix
//
// One thread per (wavelength, step): precomputes the deterministic
// eye-path context for hybrid_scatter_v2 (HCTX layout). The host binds the
// CONTEXT buffer at out_buf for this kernel.
// ============================================================================

@compute @workgroup_size(256)
fn hybrid_context_prefix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (!atm_header_valid()) {
        return; // main kernel raises the sentinel
    }
    let num_wl = atm_num_wavelengths();
    let wl_idx = tid / HYBRID_LOS_STEPS;
    let step_idx = tid % HYBRID_LOS_STEPS;
    if (wl_idx >= num_wl) {
        return;
    }

    var observer_pos = read_observer();
    let view_dir = read_view_dir();
    let secondary_rays = read_secondary_rays();
    let field_present = read_field_present();

    let toa_radius = atm_toa_radius();
    let surface_radius = atm_surface_radius();
    // Eye-path entry snap (d4f682e class): see the Metal twin.
    if (length(observer_pos) < surface_radius + BOUNDARY_NUDGE_M) {
        observer_pos = normalize(observer_pos) * (surface_radius + BOUNDARY_NUDGE_M);
    }

    // Same step geometry as hybrid_scatter_v2.
    var valid = true;
    var ds = 0.0;
    var step_start_s = 0.0;
    {
        let toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
        if (!toa_hit.hit || toa_hit.t_far <= 0.0) {
            valid = false;
        } else {
            let los_max = toa_hit.t_far;
            let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
            let hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3
                && ground_hit.t_near < los_max;
            let los_end = select(los_max, ground_hit.t_near, hits_ground);
            if (los_end <= 0.0) {
                valid = false;
            } else {
                let num_steps = min(HYBRID_LOS_STEPS, u32(los_end / 500.0) + 20u);
                if (step_idx >= num_steps) {
                    valid = false;
                } else {
                    ds = los_end / f32(num_steps);
                    step_start_s = f32(step_idx) * ds;
                }
            }
        }
    }

    let base = tid * HCTX_STRIDE;
    if (!valid) {
        out_buf[base + HCTX_K_SUB] = bitcast<f32>(0u); // marks the step invalid
        return;
    }

    var global_total_rays = read_photons_per_wl();
    if (global_total_rays == 0u) {
        global_total_rays = secondary_rays;
    }

    var k_sub = 1u;
    var sub_ds = 0.0;
    var sub_tau_cloud: array<f32, 64>;
    var sub_ray_start: array<u32, 64>;
    var sub_ray_count: array<u32, 64>;
    var tau_obs_prefix = 0.0;
    var tau_cloud_prefix = 0.0;

    let cloud_channel = atm_has_cloud_channel(field_present);
    let step_start = observer_pos + view_dir * step_start_s;
    // Exact cloud tau of the COARSE step.
    var tau_cloud_coarse = 0.0;
    if (cloud_channel) {
        tau_cloud_coarse = eye_step_cloud_tau(field_present, step_start, view_dir, ds);
    }
    if (tau_cloud_coarse > CLOUD_SUBSTEP_TAU) {
        k_sub = clamp(u32(ceil(tau_cloud_coarse / CLOUD_SUBSTEP_TAU)),
                      2u, CLOUD_MAX_SUBSTEPS);
    }
    sub_ds = ds / f32(k_sub);

    if (k_sub == 1u) {
        sub_tau_cloud[0] = tau_cloud_coarse;
        sub_ray_start[0] = 0u;
        sub_ray_count[0] = global_total_rays;
    } else {
        // Importance weights: estimated contribution of each substep.
        var sub_w: array<f32, 64>;
        var sum_w = 0.0;
        var tau_pref = 0.0;
        for (var j = 0u; j < k_sub; j++) {
            let sub_start = observer_pos
                + view_dir * (step_start_s + f32(j) * sub_ds);
            let tc = eye_step_cloud_tau(field_present, sub_start, view_dir, sub_ds);
            sub_tau_cloud[j] = tc;
            let mid = sub_start + view_dir * (sub_ds * 0.5);
            var tg = 0.0;
            let smid = shell_index_binary(length(mid));
            if (smid >= 0) {
                tg = atm[ATM_OPTICS_START
                    + (u32(smid) * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE]
                    * sub_ds;
            }
            let w = (tc + tg) * exp(-(tau_pref + 0.5 * tc));
            sub_w[j] = w;
            sum_w += w;
            tau_pref += tc;
        }
        // Allocation: n_j = round(N * w_j / sum_w), min 1, cumulatively
        // capped so the substep ranges PARTITION [0, N).
        if (global_total_rays < k_sub) {
            // Fewer rays than substeps: a disjoint partition would drop
            // whole substeps beyond the dispatch domain (order-2+ bias,
            // review round 2). Full range per substep: unbiased,
            // correlated-substeps variance only.
            for (var j = 0u; j < k_sub; j = j + 1u) {
                sub_ray_start[j] = 0u;
                sub_ray_count[j] = global_total_rays;
            }
        } else {
        var assigned = 0u;
        for (var j = 0u; j < k_sub; j++) {
            let remaining = k_sub - 1u - j;
            var nj: u32;
            if (sum_w > 1e-30) {
                nj = u32(round_half_up(f32(global_total_rays) * sub_w[j] / sum_w));
            } else {
                nj = max(global_total_rays / k_sub, 1u);
            }
            nj = max(nj, 1u);
            var cap = 1u;
            if (global_total_rays > assigned + remaining) {
                cap = global_total_rays - assigned - remaining;
            }
            nj = min(nj, cap);
            sub_ray_start[j] = assigned;
            sub_ray_count[j] = nj;
            assigned += nj;
        }
        }
    }

    // Eye-path prefix optical depths to the START of this coarse step.
    // Gas: per-step midpoint quadrature; cloud: ONE exact call over the
    // whole prefix (see the MSL comment on the documented quadrature
    // deviation).
    for (var j = 0u; j < step_idx; j++) {
        let sj = (f32(j) + 0.5) * ds;
        let pj = observer_pos + view_dir * sj;
        let rj = length(pj);
        if (rj <= toa_radius && rj >= surface_radius) {
            let sj_idx = shell_index_binary(rj);
            if (sj_idx >= 0) {
                tau_obs_prefix += atm[ATM_OPTICS_START
                    + (u32(sj_idx) * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE]
                    * ds;
            }
        }
    }
    if (cloud_channel && step_idx > 0u) {
        tau_cloud_prefix = eye_step_cloud_tau(field_present, observer_pos,
                                              view_dir, step_start_s);
    }

    out_buf[base + HCTX_TAU_OBS] = tau_obs_prefix;
    out_buf[base + HCTX_TAU_CLOUD] = tau_cloud_prefix;
    out_buf[base + HCTX_K_SUB] = bitcast<f32>(k_sub);
    out_buf[base + HCTX_SPARE] = 0.0;
    for (var j = 0u; j < k_sub; j++) {
        out_buf[base + HCTX_SUB_TAU + j] = sub_tau_cloud[j];
        out_buf[base + HCTX_SUB_START + j] = bitcast<f32>(sub_ray_start[j]);
        out_buf[base + HCTX_SUB_COUNT + j] = bitcast<f32>(sub_ray_count[j]);
    }
}

// ============================================================================
// Kernel 3b: hybrid_scatter_v2 (ray-parallel)
//
// Dispatch: (num_wavelengths, num_steps) workgroups of 64 threads.
//   workgroup_id.x = wavelength index
//   workgroup_id.y = COARSE LOS step index (plus the host step-window offset)
//   local id x     = chain lane within this step
//
// See twilight.metal for the estimator documentation (substepping, global
// chain budget partition, split-dispatch contract). REDUCTION: portable
// workgroup-shared-memory baseline (all 64 lanes write, one barrier, lane
// 0 Kahan-sums), replacing Metal's simdgroup reduction: reduction order
// differs, which is irrelevant for a statistical estimator. Control flow
// is restructured so the single workgroupBarrier sits in UNIFORM control
// flow (no early returns), which WGSL requires.
// ============================================================================

var<workgroup> wg_partials: array<f32, 64>;

@compute @workgroup_size(64)
fn hybrid_scatter_v2(@builtin(workgroup_id) tg_pos: vec3<u32>,
                     @builtin(local_invocation_id) tid_in_tg: vec3<u32>) {
    let wl_idx = tg_pos.x;
    // Absolute LOS step index: workgroup y plus the host's window offset.
    let step_idx = tg_pos.y + read_step_offset();
    let ray_lane = tid_in_tg.x;

    let header_ok = atm_header_valid();
    if (!header_ok) {
        if (wl_idx == 0u && step_idx == 0u && ray_lane == 0u) {
            out_buf[0] = HEADER_SENTINEL;
        }
    }

    var my_contribution = 0.0;

    if (header_ok) {
        let num_wl = atm_num_wavelengths();

        var observer_pos = read_observer();
        let view_dir = read_view_dir();
        let sun_dir = read_sun_dir();
        let secondary_rays = read_secondary_rays();
        let field_present = read_field_present();

        let toa_radius = atm_toa_radius();
        let surface_radius = atm_surface_radius();
        // Eye-path entry snap: must match hybrid_context_prefix exactly.
        if (length(observer_pos) < surface_radius + BOUNDARY_NUDGE_M) {
            observer_pos = normalize(observer_pos) * (surface_radius + BOUNDARY_NUDGE_M);
        }

        // Uniform coarse-step geometry.
        var valid = (wl_idx < num_wl);
        var ds = 0.0;
        var step_start_s = 0.0;
        if (valid) {
            let toa_hit = ray_sphere_intersect(observer_pos, view_dir, toa_radius);
            if (!toa_hit.hit || toa_hit.t_far <= 0.0) {
                valid = false;
            } else {
                let los_max = toa_hit.t_far;
                let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
                let hits_ground = ground_hit.hit && ground_hit.t_near > 1e-3
                    && ground_hit.t_near < los_max;
                let los_end = select(los_max, ground_hit.t_near, hits_ground);
                if (los_end <= 0.0) {
                    valid = false;
                } else {
                    let num_steps = min(HYBRID_LOS_STEPS, u32(los_end / 500.0) + 20u);
                    if (step_idx >= num_steps) {
                        valid = false;
                    } else {
                        ds = los_end / f32(num_steps);
                        step_start_s = f32(step_idx) * ds;
                    }
                }
            }
        }

        // Global chain budget (stratification + substep allocation domain).
        var global_total_rays = read_photons_per_wl();
        if (global_total_rays == 0u) {
            global_total_rays = secondary_rays;
        }
        // ray_offset from the upper 32 bits of the seed (split-dispatch).
        let raw_seed = read_rng_seed();
        let ray_offset = raw_seed.y;
        let base_seed = vec2<u32>(raw_seed.x, 0u);
        // Per-thread RNG stream (IDENTICAL derivation to the Metal port:
        // base_seed ^ wl*GAMMA ^ step<<16 ^ (lane+offset)<<32, splitmix64).
        let seed_input = base_seed
            ^ u64_mul(vec2<u32>(wl_idx, 0u), SM64_GAMMA)
            ^ u64_shl(vec2<u32>(step_idx, 0u), 16u)
            ^ vec2<u32>(0u, ray_lane + ray_offset);
        var rng = splitmix64(seed_input);

        // Precomputed eye-path context (hybrid_context_prefix).
        var cloud_channel = false;
        var k_sub = 1u;
        var sub_ds = 0.0;
        var tau_obs_prefix = 0.0;
        var tau_cloud_prefix = 0.0;
        let ctx_base = (wl_idx * HYBRID_LOS_STEPS + step_idx) * HCTX_STRIDE;
        if (valid) {
            cloud_channel = atm_has_cloud_channel(field_present);
            k_sub = bitcast<u32>(hctx[ctx_base + HCTX_K_SUB]);
            if (k_sub == 0u) {
                valid = false;
            } else {
                sub_ds = ds / f32(k_sub);
                tau_obs_prefix = hctx[ctx_base + HCTX_TAU_OBS];
                tau_cloud_prefix = hctx[ctx_base + HCTX_TAU_CLOUD];
            }
        }

        // Serial substep loop (only the chain loop diverges per thread, in
        // trip count; no barriers inside).
        if (valid) {
            var running_tau = tau_obs_prefix;
            var running_tau_cloud = tau_cloud_prefix;
            for (var sub = 0u; sub < k_sub; sub++) {
                let s = step_start_s + (f32(sub) + 0.5) * sub_ds;
                let scatter_pos = observer_pos + view_dir * s;
                let r = length(scatter_pos);
                if (r > toa_radius || r < surface_radius) {
                    continue;
                }
                let my_sidx = shell_index_binary(r);
                if (my_sidx < 0) {
                    continue;
                }

                let my_op = read_optics(u32(my_sidx), wl_idx);
                let my_beta_scat = my_op.extinction * my_op.ssa;
                let tau_cloud_step = hctx[ctx_base + HCTX_SUB_TAU + sub];
                let beta_cloud = tau_cloud_step / sub_ds;

                // A substep contributes when EITHER channel scatters here.
                if (my_beta_scat < 1e-30 && beta_cloud <= 0.0) {
                    running_tau += my_op.extinction * sub_ds;
                    running_tau_cloud += tau_cloud_step;
                    continue;
                }

                // Chain-mode eye transmittance: Beer-Lambert for gas AND
                // cloud (NO T_diff on chain paths).
                let t_obs = exp(-(running_tau + my_op.extinction * sub_ds * 0.5
                                  + running_tau_cloud + tau_cloud_step * 0.5));
                if (t_obs < 1e-30) {
                    break; // LOS opaque: later substeps darker still
                }

                // Local asymmetry for this substep's cloud source terms.
                var g_cloud_step = 0.0;
                if (beta_cloud > 0.0) {
                    if (field_present) {
                        g_cloud_step = field_g_at(scatter_pos);
                    } else {
                        g_cloud_step = atm[ATM_CLOUD_G_SCALED];
                    }
                }

                // Order 1: deterministic NEE, computed ONCE per run (lane 0
                // of the chunk containing global ray 0), scaled by the
                // GLOBAL ray total.
                if (ray_lane == 0u && ray_offset == 0u) {
                    let t_sun = shadow_ray_transmittance_chain(
                        field_present, cloud_channel, scatter_pos, sun_dir, wl_idx);
                    if (t_sun > 1e-30) {
                        let cos_theta_1 = dot(sun_dir, view_dir);
                        var A_1 = 0.0;
                        var B_1 = 0.0;
                        var C_1 = 0.0;
                        stokes_ABC(cos_theta_1, my_op, &A_1, &B_1, &C_1);
                        let scale_1 = my_beta_scat * INV_4PI * t_sun * t_obs * sub_ds;
                        my_contribution += A_1 * scale_1 * f32(global_total_rays);
                        // Cloud in-scatter source, order-1 NEE (gray
                        // channel): depolarizing HG, I-term only.
                        if (beta_cloud > 0.0) {
                            let scale_c = beta_cloud
                                * henyey_greenstein_phase(cos_theta_1, g_cloud_step)
                                * INV_4PI * t_sun * t_obs * sub_ds;
                            my_contribution += scale_c * f32(global_total_rays);
                        }
                    }
                }

                // Orders 2+: MC chains. This substep owns global chain
                // indices [lo, lo + n_sub); this dispatch covers
                // [ray_offset, ray_offset + secondary_rays); this thread
                // takes every 64th of the intersection.
                let n_sub = bitcast<u32>(hctx[ctx_base + HCTX_SUB_COUNT + sub]);
                if (secondary_rays > 0u && n_sub > 0u) {
                    let local_up = normalize(scatter_pos);
                    let cos_sza = dot(sun_dir, local_up);
                    let sza_deg = acos(clamp(cos_sza, -1.0, 1.0)) * (180.0 / PI);
                    let bp = branch_params_for_sza(sza_deg);
                    let sza_t_et = clamp((sza_deg - ZENITH_SZA_START_DEG)
                                         / (ZENITH_SZA_FULL_DEG - ZENITH_SZA_START_DEG),
                                         0.0, 1.0);
                    var setup: SecondarySetup;
                    setup.local_up = local_up;
                    setup.term_axis_dir = terminator_axis(local_up, sun_dir, bp.tilt_rad);
                    setup.alpha_p = 1.0 - bp.zenith_frac;
                    setup.alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
                    setup.alpha_t = bp.zenith_frac * bp.term_share;
                    setup.n_zenith = bp.n_zenith;
                    setup.m_term = bp.m_term;
                    setup.alpha_et = EXP_TRANSFORM_ALPHA_MAX * sza_t_et;
                    setup.sza_deg = sza_deg;
                    // Forced mode: on at deep twilight. The 1D deck composes
                    // via the exact combined channel; a 3D field composes
                    // via the majorant-combined channel + truncated null-
                    // collision classification, which needs the v5
                    // per-shell majorants. A field packed without them
                    // keeps the analog fallback.
                    setup.use_forced = 0u;
                    if (sza_deg >= ZENITH_SZA_START_DEG
                        && (!field_present || field_has_shell_majorants())) {
                        setup.use_forced = 1u;
                    }
                    // The CPU forced_tau_min_for_sza sigmoid.
                    setup.forced_tau_min =
                        0.05 - 0.03 / (1.0 + exp(-(sza_deg - 102.0)));
                    setup.cloud_channel = select(0u, 1u, cloud_channel);
                    setup.beta_seed = beta_cloud;
                    setup.g_seed = g_cloud_step;
                    if (field_present) {
                        setup.max_bounces = HYBRID_FIELD_MAX_BOUNCES;
                    } else {
                        setup.max_bounces = HYBRID_MAX_BOUNCES;
                    }

                    // Substep scale: beta_total with the (N / n_sub) fold
                    // (see the MSL scale_m comment).
                    let scale_m = (my_beta_scat + beta_cloud) * t_obs * sub_ds
                        * (f32(global_total_rays) / f32(n_sub));

                    let lo = bitcast<u32>(hctx[ctx_base + HCTX_SUB_START + sub]);
                    let hi = lo + n_sub;
                    let glo = max(lo, ray_offset);
                    let ghi = min(hi, ray_offset + secondary_rays);
                    var mc_I = kahan_new();
                    if (glo < ghi) {
                        // First relative index >= (glo - ray_offset) owned
                        // by this lane (relative indices stripe mod 64).
                        let rel0 = glo - ray_offset;
                        let first = rel0
                            + ((ray_lane + HYBRID_V2_THREADGROUP_SIZE
                                - (rel0 % HYBRID_V2_THREADGROUP_SIZE))
                               % HYBRID_V2_THREADGROUP_SIZE);
                        for (var rel = first; rel + ray_offset < ghi;
                             rel += HYBRID_V2_THREADGROUP_SIZE) {
                            let g = rel + ray_offset;
                            let chain = trace_secondary_chain(
                                field_present, scatter_pos, sun_dir,
                                wl_idx, my_op, view_dir, setup,
                                g - lo, n_sub, &rng);
                            let val = chain.x * scale_m;
                            if (is_finite_f32(val)) {
                                kahan_add(&mc_I, val);
                            }
                        }
                    }
                    let mc_result = kahan_result(mc_I);
                    if (is_finite_f32(mc_result)) {
                        my_contribution += mc_result;
                    }
                }

                running_tau += my_op.extinction * sub_ds;
                running_tau_cloud += tau_cloud_step;
            }
        }
    }

    // Guard: zero out non-finite contributions.
    if (!is_finite_f32(my_contribution)) {
        my_contribution = 0.0;
    }

    // Portable workgroup reduction (baseline; see kernel header).
    wg_partials[ray_lane] = my_contribution;
    workgroupBarrier();

    if (ray_lane == 0u && header_ok) {
        var final_sum = kahan_new();
        for (var i = 0u; i < HYBRID_V2_THREADGROUP_SIZE; i++) {
            let v = wg_partials[i];
            if (is_finite_f32(v)) {
                kahan_add(&final_sum, v);
            }
        }
        let result = kahan_result(final_sum);
        var out_val = 0.0;
        if (is_finite_f32(result)) {
            out_val = result;
        }
        out_buf[wl_idx * HYBRID_LOS_STEPS + step_idx] = out_val;
    }
}

// ============================================================================
// Kernel 3f: field_tau_probe (G-DDA-PARITY)
//
// One thread per ray; integrates cloud optical depth via the device-side
// field DDA. rays buffer: one 8-f32 header (slot 0 = ray count bit
// pattern), then 8 f32 per ray: (p0.xyz, t_max), (dir.xyz, pad).
// ============================================================================

@compute @workgroup_size(64)
fn field_tau_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let n_rays = bitcast<u32>(rays[0]);
    if (tid >= n_rays) {
        return;
    }
    // Header gate on the field buffer (mirrors the atmosphere gate).
    if (bitcast<u32>(fld[FIELD_HDR_MAGIC]) != BUFFER_MAGIC
        || bitcast<u32>(fld[FIELD_HDR_VERSION]) != BUFFER_VERSION) {
        if (tid == 0u) {
            out_buf[0] = HEADER_SENTINEL;
        }
        return;
    }
    let base = 8u + tid * 8u;
    let p0 = vec3<f32>(rays[base], rays[base + 1u], rays[base + 2u]);
    let tmx = rays[base + 3u];
    let dir = vec3<f32>(rays[base + 4u], rays[base + 5u], rays[base + 6u]);
    out_buf[tid] = field_tau_along(p0, dir, tmx);
}

// ============================================================================
// Kernel 4: garstang_zenith
//
// One thread per light source. Sources: 8 f32 each; config: 8 f32.
// ============================================================================

@compute @workgroup_size(256)
fn garstang_zenith(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;

    // Read config.
    let observer_elevation = gcfg[0];
    let aod_550 = gcfg[1];
    let uplight_fraction = gcfg[2];
    let ground_reflectance = gcfg[3];
    let wavelength_nm = gcfg[4];
    let altitude_steps = u32(gcfg[5]);
    let max_altitude = gcfg[6];
    let num_sources = u32(gcfg[7]);

    if (tid >= num_sources) {
        return;
    }

    // Read this source (8 f32 per source).
    let base = tid * 8u;
    let distance_m = gsrc[base];
    let source_rad = gsrc[base + 2u];

    if (distance_m < 1.0) {
        out_buf[tid] = 0.0;
        return;
    }

    // Rayleigh optical depth at this wavelength: lambda^-4 scaling.
    let wl_ratio = 550.0 / wavelength_nm;
    let rayleigh_tau = TAU_RAYLEIGH_550 * wl_ratio * wl_ratio * wl_ratio * wl_ratio;

    // Aerosol optical depth: Angstrom exponent ~1.3.
    let aerosol_tau = aod_550 * pow(wl_ratio, 1.3);

    let effective_up = uplight_fraction + ground_reflectance * 0.5;
    let source_intensity = source_rad * effective_up;

    let dh = max_altitude / f32(altitude_steps);
    let d = distance_m;

    var integral = kahan_new();

    for (var step_i = 0u; step_i < altitude_steps; step_i++) {
        let h = (f32(step_i) + 0.5) * dh;
        if (h < observer_elevation) {
            continue;
        }

        let r_src_to_scat = sqrt(d * d + h * h);
        let theta_scatter = PI - atan(d / max(h, 1e-6));

        // Scattering coefficients at this altitude.
        let n_rayleigh = rayleigh_tau / H_RAYLEIGH * exp(-h / H_RAYLEIGH);
        let n_aerosol = aerosol_tau / H_AEROSOL * exp(-h / H_AEROSOL);
        let sigma_total = n_rayleigh + n_aerosol;

        // Phase functions.
        let cos_scatter = cos(theta_scatter);
        let p_rayleigh = 3.0 / (16.0 * PI) * (1.0 + cos_scatter * cos_scatter);
        var p_mie = 0.0;
        {
            let g = 0.7;
            let g2 = g * g;
            let denom = 1.0 + g2 - 2.0 * g * cos_scatter;
            p_mie = (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
        }

        var f_rayleigh = 0.5;
        if (sigma_total > 0.0) {
            f_rayleigh = n_rayleigh / sigma_total;
        }
        let p_avg = f_rayleigh * p_rayleigh + (1.0 - f_rayleigh) * p_mie;

        // Slant optical depth from source to scatter point.
        let path_len = r_src_to_scat;
        var tau_slant = 0.0;
        if (h > 1.0) {
            let n0_r = rayleigh_tau / H_RAYLEIGH;
            tau_slant += n0_r * path_len * H_RAYLEIGH / h * (1.0 - exp(-h / H_RAYLEIGH));
            let n0_a = aerosol_tau / H_AEROSOL;
            tau_slant += n0_a * path_len * H_AEROSOL / h * (1.0 - exp(-h / H_AEROSOL));
        } else {
            tau_slant = (rayleigh_tau / H_RAYLEIGH + aerosol_tau / H_AEROSOL) * path_len;
        }

        // Vertical optical depth from scatter point to observer.
        let tau_vert = rayleigh_tau
            * (exp(-observer_elevation / H_RAYLEIGH) - exp(-h / H_RAYLEIGH))
            + aerosol_tau * (exp(-observer_elevation / H_AEROSOL) - exp(-h / H_AEROSOL));

        let extinction = exp(-tau_slant - tau_vert);
        let r2 = r_src_to_scat * r_src_to_scat;

        let di = source_intensity / (4.0 * PI * r2) * sigma_total * p_avg * extinction * dh;
        kahan_add(&integral, di);
    }

    out_buf[tid] = kahan_result(integral);
}
