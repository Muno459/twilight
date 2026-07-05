// Twilight MCRT - Metal Shading Language compute kernels (v2)
//
// Four compute kernels for GPU-accelerated twilight radiative transfer:
//   1. single_scatter_spectrum   - Deterministic LOS integration
//   2. mcrt_trace_photon         - Backward MC with next-event estimation
//   3. hybrid_scatter_v2         - LOS + secondary MC chains (ray-parallel)
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
// Chain bounce cap. The CPU reference runs BOUNCE_SAFETY_LIMIT = 10000
// (effectively unbiased: chains terminate by escape or f32 weight decay,
// never by this backstop); the old GPU cap of 50 was invisible in clear sky
// (chains die by ssa and escape within a few bounces) but truncates real
// energy under a conservative cloud deck: a cloud collision never shrinks
// the weight, so photons diffusing through an OD-2+ deck routinely exceed
// 50 scatters. 2000 keeps the loop bounded for the macOS GPU watchdog while
// the truncated tail (random-walk escape-time tail at tau* <= 10) sits far
// below the MC noise floor.
constant uint HYBRID_MAX_BOUNCES = 2000;
// Chain bounce cap under a 3D FIELD. Every bounce there pays a field-DDA
// shadow ray (~ms), so the 2000-bounce cap put worst-case command buffers
// far past the macOS GPU watchdog (measured: ImpactingInteractivity kills
// on the real Padborg field at watchdog batch 4, then a cooldown window
// fast-kills the pipeline). 400 keeps the worst buffer under the watchdog
// at batch 4 and is numerically negligible: for a delta-scaled deck of
// tau* <= 10, in-deck survival past n scatters decays like
// exp(-n pi^2 / (4 tau*^2)), i.e. ~5e-5 of chain weight past 400 scatters
// at tau* = 10 (real retrieved fields sit at tau* of a few: ~e-100), far
// below the MC noise floor at any gate budget. The 1D shell deck keeps the
// full 2000 cap: its per-bounce cost has no DDA and its buffers fit.
constant uint HYBRID_FIELD_MAX_BOUNCES = 400;

// Per-(wavelength, step) eye-path context, precomputed ONCE per hybrid run
// by hybrid_context_prefix and read by every hybrid_scatter_v2 chunk. The
// context (gas/cloud prefix optical depths, substep split, substep cloud
// taus, importance-allocated chain budgets) is deterministic per geometry;
// recomputing it redundantly in every watchdog-sized ray chunk made the
// field path spend ~50x its chain work on context DDAs and saturated the
// GPU into interactivity kills (the same reason the retired
// hybrid_los_prefix kernel existed for the gas prefix).
constant uint HCTX_TAU_OBS   = 0;         // gas tau, observer -> step start
constant uint HCTX_TAU_CLOUD = 1;         // cloud tau, observer -> step start
constant uint HCTX_K_SUB     = 2;         // substep count [uint bits], 0 = invalid step
constant uint HCTX_SPARE     = 3;
constant uint HCTX_SUB_TAU   = 4;         // 64 x per-substep cloud tau
constant uint HCTX_SUB_START = 4 + 64;    // 64 x global chain range start [uint bits]
constant uint HCTX_SUB_COUNT = 4 + 128;   // 64 x global chain count [uint bits]
constant uint HCTX_STRIDE    = 4 + 192;   // 196 f32 per (wl, step)
// The hybrid v2 (ray-parallel) kernel uses 64 threads to stay within
// Metal's per-threadgroup stack limit -- each trace_secondary_chain needs
// ~2-4 KB stack per thread.
constant uint HYBRID_V2_THREADGROUP_SIZE = 64;
constant uint SIMD_WIDTH = 32;
constant uint HYBRID_V2_NUM_SIMD_GROUPS = HYBRID_V2_THREADGROUP_SIZE / SIMD_WIDTH; // 2

// The hybrid eye-path quadrature approximates INT source(s) e^{-tau(s)} ds
// over a step by source(mid) * e^{-tau_mid} * ds. With a per-step cloud tau
// of x this carries a factor e^{-x/2} / [(1 - e^{-x})/x]: 0.98 at x = 0.25
// but 0.18 at x = 7.5 (a 750 m step through a tau* = 10 deck). Steps whose
// cloud tau exceeds this bound are subdivided so the midpoint rule stays
// within ~0.03 percent (port of the CPU CLOUD_SUBSTEP_TAU).
constant float CLOUD_SUBSTEP_TAU = 0.25f;
// Cap on quadrature substeps per coarse LOS step (bounds cost and the
// per-thread arrays; port of the CPU CLOUD_MAX_SUBSTEPS).
constant uint CLOUD_MAX_SUBSTEPS = 64;

// Buffer header. buffers.rs packs the u32 magic/version words bit-for-bit
// into f32 slots via f32::from_bits, so the shader compares bit patterns
// with as_type<uint>, never float values (the version word is a denormal).
constant uint ATM_HEADER_MAGIC          = 0;
constant uint ATM_HEADER_VERSION        = 1;
constant uint BUFFER_MAGIC              = 0x544C5754u; // "TWLT"
constant uint BUFFER_VERSION            = 5u;
// Written to output[0] when the header gate fails. Radiance and per-photon
// weights are never negative, so the host detects this unambiguously.
constant float HEADER_SENTINEL          = -1.0f;
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

// Early-exit threshold for the forced-scattering scout
// (scout_with_vspg_segments). At tau > 20, 1-exp(-20) = 0.999999998 in f32.
// The forced-scattering weight is indistinguishable from 1.0 and we fall
// back to analog scatter.
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

// ── Deep-twilight guiding stack (port of the CPU secondary chains) ─────────
//
// VSPG (vertical-shell-probability guiding): importance-weights the forced
// collision location across the scout's per-shell tau segments so deep-
// twilight chains scatter high (stratosphere/mesosphere) where NEE toward
// the sun still sees light. Exactly unbiased: the weight correction
// I_avg / I_j compensates the biased segment selection (CPU photon.rs,
// scout_with_vspg_segments / vspg_sample_from_segments).
//
// GPU DIFFERENCE, documented honestly: the CPU keeps
// VSPG_MAX_SEGMENTS = 128 (a full 64-shell double crossing plus
// reflection headroom). Per-thread GPU storage is register/stack-limited
// (64-thread groups, ~2-4 KB stack per thread), so the GPU cap is 64
// segments = MAX_SHELLS: a full one-way crossing is captured exactly and
// only reflection-multiplied walks overflow. Overflow uses the SAME
// tile-to-tau_max rule as the CPU (extend the LAST segment across the
// overflow tau at neutral importance 1.0): the segment set keeps tiling
// [0, tau_max], p_sum telescopes to 1 - e^{-tau_max}, and the estimator
// stays exactly unbiased with merely coarser importance on the tail.
// (No overflow diagnostic counter on GPU; the CPU atomic is
// observability-only.)
constant uint VSPG_GPU_MAX_SEGMENTS = 64;
// Altitude ramp of the VSPG importance (CPU constants, photon.rs).
constant float VSPG_BOOST_START_M = 15000.0f;
constant float VSPG_BOOST_FULL_M  = 70000.0f;
constant float VSPG_MAX_IMPORTANCE = 50.0f;
// SZA ramp of the VSPG importance.
constant float VSPG_SZA_START = 93.0f;
constant float VSPG_SZA_FULL  = 106.0f;

// Dwivedi-type horizontal direction MIS (CPU constants, photon.rs): at
// deep twilight chains must travel ~1500 km laterally; the Dwivedi lobe
// p(cos_z) ~ exp(-beta |cos_z|) concentrates direction samples near the
// local horizontal plane, mixed with the phase function via one-sample
// balance-heuristic MIS (exactly unbiased).
constant float DWIVEDI_BETA_MAX   = 3.0f;
constant float DWIVEDI_SZA_CENTER = 103.0f;
constant float DWIVEDI_SZA_WIDTH  = 2.0f;
constant float DWIVEDI_FRAC_MAX   = 0.35f;

// Iteration backstop for the field-forced truncated null-collision loop
// (CPU FIELD_NULL_EVENT_LIMIT): the expected null count per forced flight
// is bounded by the majorant-excess tau < FORCED_TAU_CUTOFF = 20, so
// P(> 512 events) is a Poisson(20) tail; a chain hitting the limit is
// killed with weight zero (expectation loss far below f32 resolution).
constant uint FIELD_NULL_EVENT_LIMIT = 512;

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

// Magic/version gate: every kernel that takes the atmosphere buffer must
// call this before reading any payload word and abort with HEADER_SENTINEL
// on mismatch, so stale or mispacked buffers fail loudly instead of
// producing plausible-looking garbage.
inline bool atm_header_valid(device const float* atm) {
    return as_type<uint>(atm[ATM_HEADER_MAGIC]) == BUFFER_MAGIC
        && as_type<uint>(atm[ATM_HEADER_VERSION]) == BUFFER_VERSION;
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

inline float read_refractive_index(device const float* atm, uint shell_idx) {
    return atm[ATM_REFRACTIVE_INDEX_START + shell_idx];
}

inline float read_cloud_extinction(device const float* atm, uint shell_idx) {
    return atm[ATM_CLOUD_EXT_START + shell_idx];
}

// Eddington diffuse transmittance of accumulated (delta-scaled) cloud
// optical depth: T = 1/(1 + 0.75 tau (1 - g*)). Mirrors the CPU's
// AtmosphereModel::cloud_diffuse_transmittance - a diffusing deck
// transmits ~20-50%, which Beer-Lambert misrepresents by orders of
// magnitude (single-representation cloud transport).
inline float cloud_diffuse_transmittance(device const float* atm, float tau_cloud) {
    if (tau_cloud <= 0.0f) return 1.0f;
    float g = atm[ATM_CLOUD_G_SCALED];
    return 1.0f / (1.0f + 0.75f * tau_cloud * (1.0f - g));
}

// True when ANY gray cloud channel is active for a run: a 3D field is bound,
// or the 1D per-shell cloud extinction is nonzero somewhere. Port of the CPU
// has_cloud_channel. Selects the chain-mode Beer-Lambert shadow path (the
// clear-sky fast path is value-identical there but skips the per-shell cloud
// reads).
inline bool atm_has_cloud_channel(device const float* atm, bool field_present) {
    if (field_present) return true;
    uint ns = atm_num_shells(atm);
    for (uint s = 0; s < ns; s++) {
        if (read_cloud_extinction(atm, s) > 0.0f) return true;
    }
    return false;
}

// ============================================================================
// 3D cloud field (v4): device-side voxel accessor + DDA, a bit-for-bit port
// of twilight-core/src/cloud_field.rs (sigma_at, g_at, tau_along,
// advance_to_tau, distance_to_next_boundary). The field runs in f32, the
// same regime as the rest of the shader; the f32 tau / crossing-root error
// budget is derived in the GPU test module (G-F32-BUDGET).
//
// Header layout mirrors buffers.rs::field_offsets.
// ============================================================================

constant uint FIELD_HDR_MAGIC      = 0u;
constant uint FIELD_HDR_VERSION    = 1u;
constant uint FIELD_NZ             = 2u;
constant uint FIELD_NLAT           = 3u;
constant uint FIELD_NLON           = 4u;
constant uint FIELD_TILE           = 5u;
constant uint FIELD_NTLAT          = 6u;
constant uint FIELD_NTLON          = 7u;
constant uint FIELD_G_STAR_PRESENT = 8u;
constant uint FIELD_BG_PRESENT     = 9u;
constant uint FIELD_MACRO_PRESENT  = 10u;
// v5: per-transport-shell cloud majorant array (host-computed via
// Cloud3DField::band_max_sigma over each shell band; the field-forced
// mode's majorant-combined channel). Present flag + start offset.
constant uint FIELD_SHELL_MAJ_PRESENT = 11u;
constant uint FIELD_Z0_M           = 12u;
constant uint FIELD_DZ_M           = 13u;
constant uint FIELD_LAT0_DEG       = 14u;
constant uint FIELD_LON0_DEG       = 15u;
constant uint FIELD_DLAT_DEG       = 16u;
constant uint FIELD_DLON_DEG       = 17u;
constant uint FIELD_G_DEFAULT      = 18u;
constant uint FIELD_SHELL_MAJ_OFFSET = 19u;
constant uint FIELD_SIGMA_OFFSET   = 20u;
constant uint FIELD_G_STAR_OFFSET  = 21u;
constant uint FIELD_MACRO_OFFSET   = 22u;
constant uint FIELD_BG_OFFSET      = 23u;

constant float DEG_TO_RAD = PI / 180.0f;

// core-compatible rem_euclid (matches cloud_field.rs::rem_euclid).
inline float field_rem_euclid(float x, float y) {
    float r = fmod(x, y);
    return (r < 0.0f) ? r + y : r;
}

struct FieldCoords { float r; float lat; float lon; };

inline FieldCoords field_sphere_coords(float3 p) {
    float r = length(p);
    float lat = asin(clamp(p.z / r, -1.0f, 1.0f)) / DEG_TO_RAD;
    float lon = atan2(p.y, p.x) / DEG_TO_RAD;
    return FieldCoords{r, lat, lon};
}

inline uint field_uint(device const float* fld, uint slot) {
    return uint(fld[slot]);
}

inline uint field_array_offset(device const float* fld, uint slot) {
    return as_type<uint>(fld[slot]);
}

inline float field_z_top_m(device const float* fld) {
    return fld[FIELD_Z0_M] + fld[FIELD_DZ_M] * float(field_uint(fld, FIELD_NZ));
}

// True when the v5 per-shell majorant array is packed (the gate for the
// field-forced mode; hosts that packed without an atmosphere leave it off
// and field runs stay analog).
inline bool field_has_shell_majorants(device const float* fld) {
    return field_uint(fld, FIELD_SHELL_MAJ_PRESENT) == 1u;
}

// Per-transport-shell cloud-extinction majorant (v5): bounds the field's
// sigma_at pointwise over shell `shell_idx`'s radial band. Only valid when
// field_has_shell_majorants().
inline float field_shell_majorant(device const float* fld, uint shell_idx) {
    return fld[field_array_offset(fld, FIELD_SHELL_MAJ_OFFSET) + shell_idx];
}

// Returns true and writes (iz, ilat, ilon) when inside the footprint.
inline bool field_indices(device const float* fld, float r, float lat, float lon,
                          thread uint &iz, thread uint &ilat, thread uint &ilon) {
    float z0 = fld[FIELD_Z0_M];
    float dz = fld[FIELD_DZ_M];
    float z = r - EARTH_RADIUS_M;
    if (z < z0 || z >= field_z_top_m(fld)) return false;
    uint nz = field_uint(fld, FIELD_NZ);
    uint nlat = field_uint(fld, FIELD_NLAT);
    uint nlon = field_uint(fld, FIELD_NLON);
    float fiz = (z - z0) / dz;
    float flat = (lat - fld[FIELD_LAT0_DEG]) / fld[FIELD_DLAT_DEG];
    float dlon = field_rem_euclid(lon - fld[FIELD_LON0_DEG], 360.0f);
    float flon = dlon / fld[FIELD_DLON_DEG];
    if (flat < 0.0f || flon < 0.0f) return false;
    uint ila = uint(flat);
    uint ilo = uint(flon);
    if (ila >= nlat || ilo >= nlon) return false;
    iz = min(uint(fiz), nz - 1u);
    ilat = ila;
    ilon = ilo;
    return true;
}

// Cloud scattering extinction [1/m] at an ECEF position (sigma_at).
inline float field_sigma_at(device const float* fld, float3 p) {
    FieldCoords c = field_sphere_coords(p);
    uint iz, ilat, ilon;
    uint nlat = field_uint(fld, FIELD_NLAT);
    uint nlon = field_uint(fld, FIELD_NLON);
    if (field_indices(fld, c.r, c.lat, c.lon, iz, ilat, ilon)) {
        uint sigma_off = field_array_offset(fld, FIELD_SIGMA_OFFSET);
        return fld[sigma_off + (iz * nlat + ilat) * nlon + ilon];
    }
    // Outside the footprint: background column (or 0 outside z range).
    float z0 = fld[FIELD_Z0_M];
    float dz = fld[FIELD_DZ_M];
    float z = c.r - EARTH_RADIUS_M;
    uint nz = field_uint(fld, FIELD_NZ);
    if (z < z0 || z >= field_z_top_m(fld) || field_uint(fld, FIELD_BG_PRESENT) == 0u) {
        return 0.0f;
    }
    uint iz_bg = min(uint((z - z0) / dz), nz - 1u);
    uint bg_off = field_array_offset(fld, FIELD_BG_OFFSET);
    return fld[bg_off + iz_bg];
}

// Asymmetry g* at a position (g_at).
inline float field_g_at(device const float* fld, float3 p) {
    if (field_uint(fld, FIELD_G_STAR_PRESENT) == 0u) {
        return fld[FIELD_G_DEFAULT];
    }
    FieldCoords c = field_sphere_coords(p);
    uint iz, ilat, ilon;
    uint nlat = field_uint(fld, FIELD_NLAT);
    uint nlon = field_uint(fld, FIELD_NLON);
    if (field_indices(fld, c.r, c.lat, c.lon, iz, ilat, ilon)) {
        uint g_off = field_array_offset(fld, FIELD_G_STAR_OFFSET);
        return fld[g_off + (iz * nlat + ilat) * nlon + ilon];
    }
    return fld[FIELD_G_DEFAULT];
}

inline float field_min_step(device const float* fld) {
    float dz = fld[FIELD_DZ_M];
    float dxy = fld[FIELD_DLAT_DEG] * DEG_TO_RAD * EARTH_RADIUS_M;
    return min(dz, dxy) * 0.25f;
}

// Macrocell majorant (max sigma) for the tile containing p. Returns 0.0 for
// a provably EMPTY tile (crossable in one coarse step, tau += 0), a positive
// value for an OCCUPIED tile (step finely), and -1.0 when p is outside the
// footprint / z range (sigma is then altitude-only, so a footprint-capped
// coarse segment has constant sigma). Mirrors the CPU macro_majorant_at.
// Only ever evaluated at SEGMENT MIDPOINTS by field_next_segment: landing
// points sit exactly on boundaries where fp parity picks an arbitrary side.
// The no-majorant-table case is handled by the CALLER (has_macro flag read
// once per traversal), so -1.0 here unambiguously means outside-footprint;
// a majorant-less in-footprint field integrates FINELY, never radial-only
// (the old traversal conflated MACRO_PRESENT == 0 with outside-footprint
// and integrated the whole field radial-only).
inline float field_macro_majorant_at(device const float* fld, float3 p) {
    if (field_uint(fld, FIELD_MACRO_PRESENT) == 0u) return -1.0f;
    FieldCoords c = field_sphere_coords(p);
    uint iz, ilat, ilon;
    if (!field_indices(fld, c.r, c.lat, c.lon, iz, ilat, ilon)) return -1.0f;
    uint tile = field_uint(fld, FIELD_TILE);
    uint ntlat = field_uint(fld, FIELD_NTLAT);
    uint ntlon = field_uint(fld, FIELD_NTLON);
    uint off = field_array_offset(fld, FIELD_MACRO_OFFSET);
    uint itlat = ilat / tile;
    uint itlon = ilon / tile;
    return fld[off + (iz * ntlat + itlat) * ntlon + itlon];
}

// Distance along dir from p to the nearest crossing of one of the SIX
// bounding surfaces of the voxel grid: the lat0 and lat0 + nlat*dlat cones,
// the lon0 and lon0 + nlon*dlon meridian planes, and the spheres at z0 and
// z_top. Caps every coarse step (empty-tile skip and out-of-footprint
// segment) so no coarse segment straddles the footprint edge or the
// z0/z_top handover. Port of the CPU distance_to_footprint_boundary: all
// six surfaces are fixed (absolute indices), so no floor-window is needed;
// the ~0 root of a just-landed-on surface is rejected by the same t > 1e-6
// guard as the lattice functions. The meridian planes are full great-circle
// planes (both halves): an antipodal-half crossing just splits a segment in
// two, which is always safe.
inline float field_distance_to_footprint_boundary(device const float* fld, float3 p, float3 dir) {
    float best = FLT_MAX;
    float z0 = fld[FIELD_Z0_M];
    float lat0 = fld[FIELD_LAT0_DEG];
    float lon0 = fld[FIELD_LON0_DEG];
    float nlat = float(field_uint(fld, FIELD_NLAT));
    float nlon = float(field_uint(fld, FIELD_NLON));
    float dlat = fld[FIELD_DLAT_DEG];
    float dlon = fld[FIELD_DLON_DEG];

    // Spheres at z0 and z_top.
    float r = length(p);
    float b_r = dot(p, dir);
    float zs[2] = { z0, field_z_top_m(fld) };
    for (uint i = 0; i < 2; i++) {
        float rk = EARTH_RADIUS_M + zs[i];
        float cc = r * r - rk * rk;
        float disc = b_r * b_r - cc;
        if (disc >= 0.0f) {
            float s = sqrt(disc);
            float t0 = -b_r - s;
            float t1 = -b_r + s;
            if (t0 > 1e-6f && t0 < best) best = t0;
            if (t1 > 1e-6f && t1 < best) best = t1;
        }
    }
    // Latitude cones at lat0 and lat0 + nlat*dlat.
    float lats[2] = { lat0, lat0 + nlat * dlat };
    for (uint i = 0; i < 2; i++) {
        float phi = lats[i] * DEG_TO_RAD;
        float tp = tan(phi);
        float a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
        float bq = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
        float cc = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
        if (fabs(a) > 1e-30f) {
            float disc = bq * bq - a * cc;
            if (disc >= 0.0f) {
                float s = sqrt(disc);
                float roots[2] = { (-bq - s) / a, (-bq + s) / a };
                for (uint j = 0; j < 2; j++) {
                    float t = roots[j];
                    if (t > 1e-6f && t < best) {
                        // Reject the mirror cone (opposite hemisphere).
                        float zc = p.z + t * dir.z;
                        if (zc * phi >= -1e-9f) best = t;
                    }
                }
            }
        } else if (fabs(bq) > 1e-30f) {
            float t = -cc / (2.0f * bq);
            if (t > 1e-6f && t < best) best = t;
        }
    }
    // Meridian planes at lon0 and lon0 + nlon*dlon.
    float lons[2] = { lon0, lon0 + nlon * dlon };
    for (uint i = 0; i < 2; i++) {
        float lam = lons[i] * DEG_TO_RAD;
        float sl = sin(lam);
        float cl = cos(lam);
        float denom = dir.x * sl - dir.y * cl;
        if (fabs(denom) > 1e-30f) {
            float t = -(p.x * sl - p.y * cl) / denom;
            if (t > 1e-6f && t < best) best = t;
        }
    }
    return best;
}

// Distance to the nearest COARSE (macro-tile) boundary: the radial z-grid is
// unchanged, but latitude/longitude crossings use the tile spacing. Used to
// skip provably-empty tiles in one step. Conservative (a smaller distance is
// always safe), so the floor-1..floor+2 window is kept.
inline float field_distance_to_next_tile_boundary(device const float* fld, float3 p, float3 dir) {
    FieldCoords c = field_sphere_coords(p);
    float r = c.r;
    float best = FLT_MAX;

    float z0 = fld[FIELD_Z0_M];
    float dz = fld[FIELD_DZ_M];
    float lat0 = fld[FIELD_LAT0_DEG];
    float lon0 = fld[FIELD_LON0_DEG];
    float tile = float(field_uint(fld, FIELD_TILE));
    float dlat_t = fld[FIELD_DLAT_DEG] * tile;
    float dlon_t = fld[FIELD_DLON_DEG] * tile;

    // Radial (fine z-grid: sigma varies per level, so keep z crossings fine).
    float z = r - EARTH_RADIUS_M;
    float iz = floor((z - z0) / dz);
    float ks_r[4] = { iz - 1.0f, iz, iz + 1.0f, iz + 2.0f };
    float b_r = dot(p, dir);
    for (uint i = 0; i < 4; i++) {
        float rk = EARTH_RADIUS_M + z0 + ks_r[i] * dz;
        float cc = r * r - rk * rk;
        float disc = b_r * b_r - cc;
        if (disc >= 0.0f) {
            float s = sqrt(disc);
            float t0 = -b_r - s;
            float t1 = -b_r + s;
            if (t0 > 1e-6f && t0 < best) best = t0;
            if (t1 > 1e-6f && t1 < best) best = t1;
        }
    }
    // Latitude cones at tile spacing.
    float flat = (c.lat - lat0) / dlat_t;
    float kf = floor(flat);
    float ks_lat[4] = { kf - 1.0f, kf, kf + 1.0f, kf + 2.0f };
    for (uint i = 0; i < 4; i++) {
        float phi = (lat0 + ks_lat[i] * dlat_t) * DEG_TO_RAD;
        float tp = tan(phi);
        float a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
        float b = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
        float cc = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
        if (fabs(a) > 1e-30f) {
            float disc = b * b - a * cc;
            if (disc >= 0.0f) {
                float s = sqrt(disc);
                float roots[2] = { (-b - s) / a, (-b + s) / a };
                for (uint j = 0; j < 2; j++) {
                    float t = roots[j];
                    if (t > 1e-6f && t < best) {
                        float zc = p.z + t * dir.z;
                        if (zc * phi >= -1e-9f) best = t;
                    }
                }
            }
        } else if (fabs(b) > 1e-30f) {
            float t = -cc / (2.0f * b);
            if (t > 1e-6f && t < best) best = t;
        }
    }
    // Longitude planes at tile spacing.
    float flon = field_rem_euclid(c.lon - lon0, 360.0f) / dlon_t;
    float kn = floor(flon);
    float ks_lon[4] = { kn - 1.0f, kn, kn + 1.0f, kn + 2.0f };
    for (uint i = 0; i < 4; i++) {
        float lam = (lon0 + ks_lon[i] * dlon_t) * DEG_TO_RAD;
        float sl = sin(lam);
        float cl = cos(lam);
        float denom = dir.x * sl - dir.y * cl;
        if (fabs(denom) > 1e-30f) {
            float t = -(p.x * sl - p.y * cl) / denom;
            if (t > 1e-6f && t < best) best = t;
        }
    }
    return best;
}

// Distance to the nearest grid-cell boundary of the three families
// (radial shells, latitude cones, longitude planes). Port of
// distance_to_next_boundary, including the floor-1..=floor+2 candidate
// window that handles fp landing parity.
inline float field_distance_to_next_boundary(device const float* fld, float3 p, float3 dir) {
    FieldCoords c = field_sphere_coords(p);
    float r = c.r;
    float best = FLT_MAX;

    float z0 = fld[FIELD_Z0_M];
    float dz = fld[FIELD_DZ_M];
    float lat0 = fld[FIELD_LAT0_DEG];
    float dlat = fld[FIELD_DLAT_DEG];
    float lon0 = fld[FIELD_LON0_DEG];
    float dlon = fld[FIELD_DLON_DEG];

    // Radial (sphere) crossings.
    float z = r - EARTH_RADIUS_M;
    float iz = floor((z - z0) / dz);
    float ks_r[4] = { iz - 1.0f, iz, iz + 1.0f, iz + 2.0f };
    float b_r = dot(p, dir);
    for (uint i = 0; i < 4; i++) {
        float rk = EARTH_RADIUS_M + z0 + ks_r[i] * dz;
        float cc = r * r - rk * rk;
        float disc = b_r * b_r - cc;
        if (disc >= 0.0f) {
            float s = sqrt(disc);
            float t0 = -b_r - s;
            float t1 = -b_r + s;
            if (t0 > 1e-6f && t0 < best) best = t0;
            if (t1 > 1e-6f && t1 < best) best = t1;
        }
    }

    // Latitude (cone) crossings.
    float flat = (c.lat - lat0) / dlat;
    float kf = floor(flat);
    float ks_lat[4] = { kf - 1.0f, kf, kf + 1.0f, kf + 2.0f };
    for (uint i = 0; i < 4; i++) {
        float phi = (lat0 + ks_lat[i] * dlat) * DEG_TO_RAD;
        float tp = tan(phi);
        float a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
        float b = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
        float cc = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
        if (fabs(a) > 1e-30f) {
            float disc = b * b - a * cc;
            if (disc >= 0.0f) {
                float s = sqrt(disc);
                float roots[2] = { (-b - s) / a, (-b + s) / a };
                for (uint j = 0; j < 2; j++) {
                    float t = roots[j];
                    if (t > 1e-6f && t < best) {
                        float zc = p.z + t * dir.z;
                        if (zc * phi >= -1e-9f) best = t;
                    }
                }
            }
        } else if (fabs(b) > 1e-30f) {
            float t = -cc / (2.0f * b);
            if (t > 1e-6f && t < best) best = t;
        }
    }

    // Longitude (meridian plane) crossings.
    float flon = field_rem_euclid(c.lon - lon0, 360.0f) / dlon;
    float kn = floor(flon);
    float ks_lon[4] = { kn - 1.0f, kn, kn + 1.0f, kn + 2.0f };
    for (uint i = 0; i < 4; i++) {
        float lam = (lon0 + ks_lon[i] * dlon) * DEG_TO_RAD;
        float sl = sin(lam);
        float cl = cos(lam);
        float denom = dir.x * sl - dir.y * cl;
        if (fabs(denom) > 1e-30f) {
            float t = -(p.x * sl - p.y * cl) / denom;
            if (t > 1e-6f && t < best) best = t;
        }
    }

    return best;
}

// One traversal segment starting at parameter t: writes the constant sigma
// over [t, t_next] to sigma_out and returns t_next. Port of the CPU
// next_segment (the full algorithm derivation lives on that function in
// cloud_field.rs); both field_tau_along and field_advance_to_tau walk
// through THIS one function, so the collision inversion inverts exactly the
// integral it was normalized with.
//
// 1. FINE CANDIDATE between adjacent boundaries of the full lattice (which
//    contains every footprint surface): lies in ONE voxel, ONE macro-tile,
//    ONE z level, on ONE side of the footprint edge; needs no extra cap.
// 2. Without majorant data, integrate the fine segment (sigma at its
//    midpoint).
// 3. MIDPOINT CLASSIFICATION: the majorant is evaluated at the segment
//    MIDPOINT, never at a landing point (endpoints sit exactly on
//    boundaries, where fp rounding parity can index the just-left tile;
//    classifying there made the empty-tile skip re-fire from an
//    empty-to-occupied boundary and drop the whole occupied chord: 59.4
//    percent of tau on a boundary-aligned checkerboard ray).
// 4. maj_f > 0 (occupied tile): integrate the fine segment.
// 5. maj_f <= 0: COARSE EXTENSION to the next tile boundary, CAPPED by the
//    footprint surfaces so no coarse segment straddles the inside/outside
//    or z-range handover (an uncapped radial step misintegrated grazing
//    chords across the footprint edge; an uncapped tile skip overshot the
//    edge when nlat/nlon is not a multiple of tile). Classify the coarse
//    segment at ITS midpoint:
//    - maj_f == 0 && maj_c == 0: provably empty tile, skip it (sigma 0).
//    - maj_f < 0 && maj_c < 0: outside the footprint / z range, sigma is
//      altitude-only and constant over the capped segment: sample at the
//      coarse midpoint.
//    - Any disagreement (min_step flooring near tangencies / fp-degenerate
//      landings): fall back to the fine segment with ITS midpoint, always
//      valid.
// min_step flooring bounds the quadrature error by sigma_max * min_step per
// tangency event and can never skip a full cell (see the CPU doc).
inline float field_next_segment(device const float* fld, float3 p0, float3 dir,
                                float t, float t_max, float min_step,
                                bool has_macro, thread float &sigma_out) {
    float3 p = p0 + dir * t;
    float d_fine = max(field_distance_to_next_boundary(fld, p, dir), min_step);
    float t_fine = min(t + d_fine, t_max);
    float3 mid_fine = p0 + dir * ((t + t_fine) * 0.5f);
    if (!has_macro) {
        sigma_out = field_sigma_at(fld, mid_fine);
        return t_fine;
    }
    float maj_f = field_macro_majorant_at(fld, mid_fine);
    if (maj_f > 0.0f) {
        // Occupied tile: integrate finely within it.
        sigma_out = field_sigma_at(fld, mid_fine);
        return t_fine;
    }
    // Empty tile or outside the footprint: try the coarse extension, capped
    // by the footprint surfaces.
    float d_fp = field_distance_to_footprint_boundary(fld, p, dir);
    float d_coarse = max(min(field_distance_to_next_tile_boundary(fld, p, dir), d_fp), min_step);
    float t_coarse = min(t + d_coarse, t_max);
    float3 mid_coarse = p0 + dir * ((t + t_coarse) * 0.5f);
    float maj_c = field_macro_majorant_at(fld, mid_coarse);
    if (maj_f == 0.0f && maj_c == 0.0f) {
        // Provably empty tile: cross it in one step, tau += 0.
        sigma_out = 0.0f;
        return t_coarse;
    }
    if (maj_f < 0.0f && maj_c < 0.0f) {
        // Outside the footprint (or z range): sigma is altitude-only and
        // constant over the capped coarse segment.
        sigma_out = field_sigma_at(fld, mid_coarse);
        return t_coarse;
    }
    // Fine/coarse classification disagreement: the fine segment with its own
    // midpoint is always a valid constant-sigma span.
    sigma_out = field_sigma_at(fld, mid_fine);
    return t_fine;
}

// Exact cloud optical depth along p0 + t*dir, t in [0, t_max] (tau_along):
// repeated field_next_segment calls, each yielding one constant-sigma span
// (a fine voxel span, a whole empty macro-tile, or a footprint-capped
// out-of-footprint stretch of one radial band). Port of the CPU tau_along.
inline float field_tau_along(device const float* fld, float3 p0, float3 dir, float t_max) {
    if (t_max <= 0.0f) return 0.0f;
    float tau = 0.0f;
    float t = 0.0f;
    float min_step = field_min_step(fld);
    bool has_macro = field_uint(fld, FIELD_MACRO_PRESENT) != 0u;
    for (uint iter = 0; iter < 40000u; iter++) {
        if (t >= t_max) break;
        float sigma;
        float t_next = field_next_segment(fld, p0, dir, t, t_max, min_step, has_macro, sigma);
        tau += sigma * (t_next - t);
        t = t_next;
    }
    return tau;
}

// Inverse of field_tau_along: parameter t where the accumulated cloud tau
// reaches tau_target, or a negative sentinel if the segment ends first
// (advance_to_tau). Walks the SAME field_next_segment spans, so the
// inversion is exact by construction (forced/cloud-channel sampling must
// invert exactly the integral it was normalized with).
inline float field_advance_to_tau(device const float* fld, float3 p0, float3 dir,
                                  float t_max, float tau_target) {
    if (tau_target <= 0.0f) return 0.0f;
    float tau = 0.0f;
    float t = 0.0f;
    float min_step = field_min_step(fld);
    bool has_macro = field_uint(fld, FIELD_MACRO_PRESENT) != 0u;
    for (uint iter = 0; iter < 40000u; iter++) {
        if (t >= t_max) return -1.0f;
        float sigma;
        float t_next = field_next_segment(fld, p0, dir, t, t_max, min_step, has_macro, sigma);
        float dtau = sigma * (t_next - t);
        if (tau + dtau >= tau_target) {
            // Constant sigma within the segment: linear inversion (sigma > 0
            // whenever this fires; zero-sigma segments cannot lift tau).
            return t + (tau_target - tau) / max(sigma, 1e-30f);
        }
        tau += dtau;
        t = t_next;
    }
    return -1.0f;
}

// Dispatch params: 4 x vec4
// vec4(obs_x, obs_y, obs_z, field_present)
// vec4(view_x, view_y, view_z, pad)
// vec4(sun_x, sun_y, sun_z, pad)
// vec4(photons_bits, secondary_bits, seed_lo_bits, seed_hi_bits)
inline float3 read_observer(device const float* params) {
    return float3(params[0], params[1], params[2]);
}
inline bool read_field_present(device const float* params) {
    return params[3] != 0.0f;
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
// LOS step-window offset (slot 7, 0 when the host does not split): added to
// the threadgroup y-index so a (wl x 200) hybrid dispatch can be issued as
// several smaller command buffers along the step axis (static watchdog
// split for the field path).
inline uint read_step_offset(device const float* params) {
    return as_type<uint>(params[7]);
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

// Surface and top-of-atmosphere radii from the PACKED SHELLS - the single
// source of truth, exactly matching the CPU's AtmosphereModel accessors
// (surface_radius = r_inner of shell 0, toa_radius = r_outer of the last
// shell). The old hardcoded EARTH_RADIUS_M + TOA_ALTITUDE_M (100 km)
// silently truncated the 150 km USSA-76 thermosphere extension - the very
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
            // On-boundary degeneracy (mirrors the CPU fix): origin on
            // the inner sphere moving inward has inner t_near ~ 0;
            // without this the walk teleports through the shell below.
            float m = dot(pos, dir);
            float r2 = dot(pos, pos);
            float b2 = max(r2 - m * m, 0.0f);
            if (inner.hit && m < 0.0f && b2 < r_inner * r_inner
                && inner.t_far > EPS
                && fabs(length(pos) - r_inner) < 1.0f) {
                result.dist = 1e-4f;
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

// Chain-mode shadow ray: the gray cloud channel is delta-tracked explicitly
// in the chains, so every transmittance leg uses Beer-Lambert exp(-tau_cloud)
// with tau_cloud integrated by the field DDA over each straight in-shell
// segment (pos/dir taken BEFORE the refraction step, matching the CPU's
// cloud_tau_segment call site) when a field is bound, or accumulated from
// the per-shell 1D cloud extinction otherwise: one implementation, two
// sigma sources, exactly like the CPU
// shadow_ray_transmittance(field, CloudTransmittance::BeerLambert).
// NO T_diff anywhere on chain paths (mixing it with explicit cloud
// scattering double-counts the diffusion). Clear-sky runs (no channel) take
// the pre-existing fast path, which is value-identical there (tau_cloud is
// identically zero, so its T_diff factor is 1).
float shadow_ray_transmittance_chain(device const float* atm, device const float* fld,
                                     bool field_present, bool cloud_channel,
                                     float3 start_pos, float3 sun_dir, uint wl_idx) {
    if (!cloud_channel) {
        return shadow_ray_transmittance(atm, start_pos, sun_dir, wl_idx);
    }
    uint ns = atm_num_shells(atm);
    float surface_radius = atm[ATM_SHELLS_START];

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
    float tau = 0.0f;
    float tau_cloud = 0.0f;

    int sidx = shell_index_binary(atm, length(pos));
    if (sidx < 0) return 1.0f;
    uint us = uint(sidx);

    for (uint iter = 0; iter < 200; iter++) {
        uint shell_base = ATM_SHELLS_START + us * ATM_SHELL_STRIDE;
        float r_inner = atm[shell_base];
        float r_outer = atm[shell_base + 1];
        float extinction = atm[ATM_OPTICS_START + (us * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) break;

        tau += extinction * bnd.dist;
        // Already opaque from clear air plus the cloud crossed so far: the
        // remaining transmittance is below exp(-35) ~ 6e-16, lost in f32
        // rounding against any twilight radiance. Return BEFORE the per-shell
        // field DDA so an opaque shadow ray (every deep-twilight grazing leg)
        // skips the dominant cost for the rest of the path. Identical combined
        // threshold to the CPU trace_transmittance so the backends match.
        if (tau + tau_cloud > 35.0f) return 0.0f;
        // Cloud tau for this straight segment: pos/dir BEFORE refraction.
        // Field DDA when bound, per-shell 1D extinction otherwise.
        tau_cloud += field_present
            ? field_tau_along(fld, pos, dir, bnd.dist)
            : read_cloud_extinction(atm, us) * bnd.dist;

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
            return 0.0f;
        }
        if (next_shell >= ns) break;
        us = next_shell;
        if (tau > 50.0f) return 0.0f;
    }

    // Clear-air Beer-Lambert x cloud Beer-Lambert (explicit scattering).
    return fast::exp(-tau) * fast::exp(-tau_cloud);
}

// Total path length of the ray origin + t*dir, t in [0, t_max], inside the
// spherical shell annulus [r_inner, r_outer]. Analytic (no stepping),
// port of the CPU single_scatter::ray_path_through_shell. The DS-precision
// ray_sphere_intersect keeps the f32 interval endpoints stable at Earth
// scale.
float ray_path_through_shell(float3 origin, float3 dir,
                             float r_inner, float r_outer, float t_max) {
    RaySphereHit outer = ray_sphere_intersect(origin, dir, r_outer);
    if (!outer.hit) return 0.0f;
    float o0 = max(outer.t_near, 0.0f);
    float o1 = min(outer.t_far, t_max);
    if (o1 <= o0 + 1e-6f) return 0.0f;

    RaySphereHit inner = ray_sphere_intersect(origin, dir, r_inner);
    float i0 = 0.0f;
    float i1 = 0.0f;
    bool has_inner = false;
    if (inner.hit) {
        i0 = max(inner.t_near, 0.0f);
        i1 = min(inner.t_far, t_max);
        has_inner = (i1 > i0 + 1e-6f);
    }
    if (!has_inner) return o1 - o0;

    // Shell interval = outer interval minus inner interval: 0, 1, or 2 segments.
    float total = 0.0f;
    float seg1_end = min(o1, i0);
    if (seg1_end > o0) total += seg1_end - o0;
    float seg2_start = max(o0, i1);
    if (o1 > seg2_start) total += o1 - seg2_start;
    return total;
}

// Exact cloud optical depth of one straight eye-path (sub)step: the shared
// field DDA when a field is bound, else analytic per-shell path lengths
// through the 1D shell deck. Port of the CPU eye_step_cloud_tau, which
// replaced the previous eye-path convention (midpoint-classified shell
// sigma times the WHOLE step): a 750 m zenith step one third below a deck
// still got full-deck extinction over its entire length, inflating the
// deck's in-scatter source and the eye cloud OD by up to 1.5x.
float eye_step_cloud_tau(device const float* atm, device const float* fld,
                         bool field_present, float3 start, float3 dir, float ds) {
    if (field_present) {
        return field_tau_along(fld, start, dir, ds);
    }
    float tau = 0.0f;
    uint ns = atm_num_shells(atm);
    for (uint s = 0; s < ns; s++) {
        float sigma_c = read_cloud_extinction(atm, s);
        if (sigma_c <= 0.0f) continue;
        uint shell_base = ATM_SHELLS_START + s * ATM_SHELL_STRIDE;
        tau += sigma_c
            * ray_path_through_shell(start, dir, atm[shell_base], atm[shell_base + 1], ds);
    }
    return tau;
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
// component, NOT from truncation - so no weight clamps and no NaN-prone
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
    // Chain-local solar zenith angle [deg]: drives the VSPG importance and
    // the Dwivedi MIS ramps (port of the CPU sza_deg_local).
    float sza_deg;
    // Forced-collision gate, computed by the caller as
    // (sza >= ZENITH_SZA_START_DEG) && (!field_present || field majorants
    // packed). Forced mode composes EXACTLY with the shell-constant 1D gray
    // cloud deck (combined transport channel: scout/advance accumulate
    // gas + cloud tau, both piecewise constant per shell, so the truncated
    // inversion stays exact and the vertex type is drawn from the extinction
    // conditional at the collision shell; port of the CPU combined-channel
    // forced mode, ac673c7). A 3D FIELD's sigma_c(x) is NOT shell-constant,
    // so forced mode there folds the MAJORANT-combined channel
    // sigma_m = sigma_gas + c_maj(shell) (exactly piecewise constant) and
    // classifies the collision by truncated null-collision delta tracking
    // (real cloud / real gas / null; port of the CPU field_forced_classify;
    // derivation and telescoping proof at the CPU scalar chain's
    // use_forced block). Fields packed WITHOUT majorants (pre-v5 hosts /
    // no uploaded atmosphere) keep the analog fallback.
    uint use_forced;
    // SZA-adaptive forced-scatter tau floor (port of the CPU
    // forced_tau_min_for_sza sigmoid: 0.05 at SZA <= 100, 0.02 at >= 104).
    float forced_tau_min;
    // Any gray cloud channel active this run (3D field or 1D shells)?
    // Selects the chain-mode Beer-Lambert shadow path.
    uint cloud_channel;
    // Gray cloud scattering coefficient of the seeding LOS substep [1/m]
    // (the substep's exact cloud tau / sub_ds). Zero on clear substeps: the
    // seed type draw is then skipped so clear-sky RNG streams keep their
    // structure (port of the CPU ChainCloud::beta_seed).
    float beta_seed;
    // Bounce cap for this run: HYBRID_MAX_BOUNCES normally,
    // HYBRID_FIELD_MAX_BOUNCES under a 3D field (watchdog bound; see the
    // constants for the truncation-tail justification).
    uint max_bounces;
    // Delta-scaled asymmetry g* for a cloud-seeded vertex.
    float g_seed;
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
// Deep-twilight guiding helpers (port of the CPU photon.rs guiding stack)
// ============================================================================

// Sigmoid for smooth SZA ramps (CPU sigmoid).
inline float guide_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// SZA-adaptive Dwivedi sampling fraction (CPU dwivedi_frac).
inline float dwivedi_frac(float sza_deg) {
    return DWIVEDI_FRAC_MAX * guide_sigmoid((sza_deg - DWIVEDI_SZA_CENTER) / DWIVEDI_SZA_WIDTH);
}

// SZA-adaptive Dwivedi concentration parameter (CPU dwivedi_beta).
inline float dwivedi_beta(float sza_deg) {
    return DWIVEDI_BETA_MAX * guide_sigmoid((sza_deg - DWIVEDI_SZA_CENTER) / DWIVEDI_SZA_WIDTH);
}

// Dwivedi PDF in sr^-1 for cos_z = dir . local_up (CPU dwivedi_pdf):
//   p(cos_z) = beta * exp(-beta |cos_z|) / (4 pi (1 - exp(-beta))),
// 1/(4 pi) when beta < 1e-6 (effectively uniform).
inline float dwivedi_pdf(float cos_z, float beta) {
    if (beta < 1e-6f) return INV_4PI;
    float abs_cz = clamp(fabs(cos_z), 0.0f, 1.0f);
    return beta * exp(-beta * abs_cz) / (4.0f * PI * (1.0f - exp(-beta)));
}

// Sample the Dwivedi distribution (CPU dwivedi_sample): CDF inversion of
// |cos_z|, random sign (symmetric about the horizontal plane), uniform phi.
struct DwivediSample { float cos_z; float phi; };
inline DwivediSample dwivedi_sample(float xi1, float xi2, float xi_sign, float beta) {
    float phi = 2.0f * PI * xi2;
    if (beta < 1e-6f) {
        return DwivediSample{2.0f * xi1 - 1.0f, phi};
    }
    float one_minus_exp_neg_beta = 1.0f - exp(-beta);
    float abs_cz = clamp(-log(1.0f - xi1 * one_minus_exp_neg_beta) / beta, 0.0f, 1.0f);
    float cos_z = (xi_sign < 0.5f) ? abs_cz : -abs_cz;
    return DwivediSample{cos_z, phi};
}

// Altitude/SZA-dependent VSPG importance (CPU vspg_importance): >= 1,
// ramping quadratically in altitude from 1.0 at 15 km to an SZA-dependent
// max (up to 50x) at 70 km.
inline float vspg_importance(float alt_m, float sza_deg) {
    if (alt_m <= VSPG_BOOST_START_M) return 1.0f;
    float sza_t = clamp((sza_deg - VSPG_SZA_START) / (VSPG_SZA_FULL - VSPG_SZA_START), 0.0f, 1.0f);
    float alt_t = clamp((alt_m - VSPG_BOOST_START_M) / (VSPG_BOOST_FULL_M - VSPG_BOOST_START_M),
                        0.0f, 1.0f);
    float max_imp = 1.0f + (VSPG_MAX_IMPORTANCE - 1.0f) * sza_t;
    return 1.0f + (max_imp - 1.0f) * alt_t * alt_t;
}

// Per-shell VSPG segment: the combined tau range a shell contributes to the
// scout walk plus its precomputed importance (CPU VspgSegment).
struct VspgSegment {
    float tau_lo;
    float tau_hi;
    float importance;
};

// Per-shell cloud channel of the forced-flight scout/advance walk: the gray
// 1D deck (exact) without a field, the per-shell field MAJORANT with one
// (field runs carry all-zero atm.cloud_extinction, so the two never mix).
inline float chain_shell_cloud_ext(device const float* atm, device const float* fld,
                                   bool use_field_maj, uint shell_idx) {
    return use_field_maj ? field_shell_majorant(fld, shell_idx)
                         : read_cloud_extinction(atm, shell_idx);
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
    if (!atm_header_valid(atm)) {
        if (tid == 0) output[0] = HEADER_SENTINEL;
        return;
    }
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
    if (!atm_header_valid(atm)) {
        if (tid == 0) output[0] = HEADER_SENTINEL;
        return;
    }
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
    // Surface snap (port of the CPU trace_photon entry snap, d4f682e): an
    // observer whose radius rounds BELOW the surface lands where
    // shell_index is -1 and every photon dies on entry with exactly zero
    // radiance. In f32 the rounding granularity at Earth radius is ~0.5 m,
    // so snap to the same BOUNDARY_NUDGE_M ledge the ground bounce uses
    // (the CPU's 1 mm ledge vanishes in f32); physically negligible,
    // numerically decisive.
    {
        float r0 = length(pos);
        float ledge = surface_radius + BOUNDARY_NUDGE_M;
        if (r0 > 0.0f && r0 < ledge) {
            pos = pos * (ledge / r0);
        }
    }
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
                float3 normal = normalize(boundary_pos);
                // Ground-bounce NEE (Lambertian albedo/pi), BEFORE the
                // albedo is folded into the continuing weight - mirrors
                // the CPU chain exactly; this path family was simply
                // missing from the GPU mcrt estimator (audit 2026-06-12).
                float cos_sun_g = dot(sun_dir, normal);
                if (cos_sun_g > 0.0f) {
                    float t_sun_g = shadow_ray_transmittance(atm, boundary_pos + normal * BOUNDARY_NUDGE_M,
                                                             sun_dir, wl_idx);
                    if (t_sun_g > 1e-30f) {
                        result_weight.add(weight * albedo * t_sun_g * cos_sun_g * (1.0f / PI));
                    }
                }
                weight *= albedo;
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

        // Apply SSA for this scatter event BEFORE NEE - the vertex is a
        // scattering event, so the survival probability multiplies every
        // contribution from it (CPU convention, photon.rs SSA-before-NEE;
        // the old order overcounted NEE by 1/ssa per order).
        weight *= op.ssa;

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
// Forced scattering scout, fused with VSPG segment collection (port of the
// CPU scout_with_vspg_segments)
//
// Marches shell-by-shell with refraction, accumulating the COMBINED
// (gas + per-shell cloud channel) optical depth to the boundary AND
// collecting the per-shell VSPG segments the forced-flight sampler inverts.
// Early-exits at tau > FORCED_TAU_CUTOFF (20.0) since the weight correction
// is indistinguishable from 1.0 at that point (segments are then unused).
// Returns (tau_max, hit_ground). When hit_ground is true, forced scattering
// should NOT be used -- the analog loop handles ground reflection.
// ============================================================================

struct ScoutResult {
    float tau;
    bool  hit_ground;
};

ScoutResult scout_with_vspg_segments(device const float* atm, device const float* fld,
                                     bool use_field_maj, float3 start_pos,
                                     float3 start_dir, uint wl_idx, float sza_deg,
                                     thread VspgSegment* segments,
                                     thread uint &num_seg) {
    uint ns = atm_num_shells(atm);
    float surface_radius = atm[ATM_SHELLS_START];
    float3 pos = start_pos;
    float3 dir = start_dir;
    float tau = 0.0f;
    num_seg = 0;

    int sidx = shell_index_binary(atm, length(pos));
    if (sidx < 0) return ScoutResult{0.0f, false};
    uint us = uint(sidx);

    for (uint iter = 0; iter < 200; iter++) {
        uint shell_base = ATM_SHELLS_START + us * ATM_SHELL_STRIDE;
        float r_inner = atm[shell_base];
        float r_outer = atm[shell_base + 1];
        float alt_mid = atm[shell_base + 2];
        float extinction = atm[ATM_OPTICS_START + (us * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) return ScoutResult{tau, false};

        // COMBINED transport channel (port of the CPU scout, ac673c7): gas
        // extinction PLUS the per-shell cloud channel -- the gray 1D shell
        // deck exactly, or the per-shell FIELD MAJORANT in field-forced mode
        // (chain_shell_cloud_ext) -- both piecewise constant per shell so
        // the sum stays exactly piecewise constant. Clear-sky shells add 0.0
        // (bit-identical to the gas-only scout).
        float tau_shell = (extinction + chain_shell_cloud_ext(atm, fld, use_field_maj, us))
                        * bnd.dist;
        float tau_end = tau + tau_shell;

        // Collect the VSPG segment when the shell carries nonzero tau.
        if (num_seg < VSPG_GPU_MAX_SEGMENTS && tau_shell > 1e-30f) {
            segments[num_seg] = VspgSegment{tau, tau_end, vspg_importance(alt_mid, sza_deg)};
            num_seg++;
        } else if (tau_shell > 1e-30f) {
            // Segment-buffer OVERFLOW. The sampler normalizes by p_sum over
            // the SEGMENTS, so dropped tau would over-weight every head
            // collision and never sample the tail. Extend the LAST segment
            // across the overflow tau at neutral importance instead (any
            // positive importance is unbiased; only the TILING of
            // [0, tau_max] matters): p_sum then telescopes to
            // 1 - e^{-tau_max} exactly (CPU overflow rule; see the
            // VSPG_GPU_MAX_SEGMENTS comment for the reduced GPU cap).
            segments[VSPG_GPU_MAX_SEGMENTS - 1].tau_hi = tau_end;
            segments[VSPG_GPU_MAX_SEGMENTS - 1].importance = 1.0f;
        }

        tau = tau_end;

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
// VSPG forced-flight sampler (port of the CPU vspg_sample_from_segments)
//
// CDF-inverts the importance-weighted per-segment scatter probabilities
// q_i = I_i * (e^{-tau_lo_i} - e^{-tau_hi_i}) collected by the scout, then
// samples tau within the chosen segment from the conditional truncated
// exponential. Returns (tau_s, weight_correction = I_avg / I_j): exactly
// unbiased for ANY positive importances over a tiling of [0, tau_max].
// Falls back to the plain truncated exponential (weight 1) when no
// segments exist or all probabilities vanish, consuming ONE rng draw like
// the CPU fallback.
// ============================================================================

struct VspgSample {
    float tau_s;
    float weight;
};

VspgSample vspg_sample_from_segments(thread const VspgSegment* segments, uint num_seg,
                                     float tau_max, thread ulong &rng) {
    if (num_seg == 0) {
        float xi = xorshift_f32(rng);
        float one_minus_exp = 1.0f - exp(-tau_max);
        return VspgSample{-log(1.0f - xi * one_minus_exp + 1e-30f), 1.0f};
    }

    float p_sum = 0.0f;
    float q_sum = 0.0f;
    for (uint i = 0; i < num_seg; i++) {
        float p_i = exp(-segments[i].tau_lo) - exp(-segments[i].tau_hi);
        p_sum += p_i;
        q_sum += segments[i].importance * p_i;
    }

    if (q_sum < 1e-30f) {
        float xi = xorshift_f32(rng);
        float one_minus_exp = 1.0f - exp(-tau_max);
        return VspgSample{-log(1.0f - xi * one_minus_exp + 1e-30f), 1.0f};
    }

    // CDF inversion: select segment j. Re-accumulates the SAME running sums
    // as the pass above (identical order, identical f32 values), so this is
    // the CPU's q_cdf scan without the per-thread cdf array.
    float xi_segment = xorshift_f32(rng) * q_sum;
    uint j = 0;
    float q_run = segments[0].importance
                * (exp(-segments[0].tau_lo) - exp(-segments[0].tau_hi));
    while (j + 1 < num_seg && q_run < xi_segment) {
        j++;
        q_run += segments[j].importance
               * (exp(-segments[j].tau_lo) - exp(-segments[j].tau_hi));
    }

    // Within segment j: conditional truncated exponential,
    // tau = -ln(e^{-tau_lo} - xi * p_j), clamped for numerical safety.
    float p_j = exp(-segments[j].tau_lo) - exp(-segments[j].tau_hi);
    float xi_within = xorshift_f32(rng);
    float tau_s = -log(exp(-segments[j].tau_lo) - xi_within * p_j + 1e-30f);
    tau_s = clamp(tau_s, segments[j].tau_lo, segments[j].tau_hi);

    // Weight correction I_avg / I_j with I_avg = q_sum / p_sum: exactly
    // compensates the biased segment selection.
    float i_avg = q_sum / p_sum;
    return VspgSample{tau_s, i_avg / segments[j].importance};
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

AdvanceResult advance_to_optical_depth(device const float* atm, device const float* fld,
                                        bool use_field_maj, float3 start_pos,
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
        // Combined per-shell extinction (gas + gray 1D cloud, or gas + the
        // per-shell field MAJORANT in field-forced mode), matching the
        // combined scout so a forced flight sampled from the combined tau
        // inverts to the exact combined collision point (CPU
        // advance_to_optical_depth, ac673c7). Clear shells add 0.0.
        float sigma_comb = atm[ATM_OPTICS_START + optics_idx * ATM_OPTICS_STRIDE]
                         + chain_shell_cloud_ext(atm, fld, use_field_maj, us);

        ShellBoundary bnd = next_shell_boundary(pos, dir, r_inner, r_outer);
        if (!bnd.found) return AdvanceResult{pos, dir, us};

        float tau_shell = sigma_comb * bnd.dist;

        if (tau_acc + tau_shell >= tau_target) {
            // Scatter point is within this shell
            float tau_remaining = tau_target - tau_acc;
            float dist = (sigma_comb > 1e-30f) ? tau_remaining / sigma_comb : bnd.dist;
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
// Secondary chain tracer (used by the hybrid_scatter_v2 kernel)
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

float4 trace_secondary_chain(device const float* atm, device const float* fld,
                             bool field_present, float3 start_pos,
                             float3 sun_dir, uint wl_idx,
                             ShellOptics start_optics, float3 prev_dir_in,
                             SecondarySetup setup,
                             uint ray_idx, uint total_rays,
                             thread ulong &rng) {
    float surface_radius = atm_surface_radius(atm);

    // Unbiased one-sample-MIS seed (port of the CPU estimator): sample
    // omega from the 3-component mixture, weight by
    //   w0 = P(omega.view)/4pi / q(omega)
    // identically for every branch - the balance-heuristic estimator.
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
    // Cloud-seed mixture (port of the CPU derivation at the scalar chain
    // seed): the seeding substep's orders-2+ source is
    // beta_gas * INT P_gas/4pi L + beta_cloud * INT P_HG/4pi L. One type
    // draw selects the vertex type with p_c = beta_cloud / beta_total; the
    // selection probability cancels the per-type coefficient exactly, so
    // the caller scales the chain average by beta_total = beta_gas +
    // beta_cloud and the expectation is exact. The draw is consumed ONLY
    // when the seeding substep carries cloud, so clear-sky RNG streams keep
    // their structure; the walk after the seed is vertex-type agnostic.
    // Per-flight stream order (single GPU stream serializing the CPU's
    // dir/ctl/tau streams in event order): seed jitter + direction draws,
    // then this type draw, then the walk draws.
    bool seed_is_cloud = false;
    if (setup.beta_seed > 0.0f) {
        float beta_gas_seed = start_optics.extinction * start_optics.ssa;
        float p_c = setup.beta_seed / (setup.beta_seed + beta_gas_seed);
        seed_is_cloud = xorshift_f32(rng) < p_c;
    }
    // prev_dir_in is the LOS view direction: the physical seed-scatter
    // cosine is omega.view (matches the CPU convention). A cloud seed swaps
    // the numerator phase to the gray HG lobe against the SAME q.
    float phase_seed = seed_is_cloud
        ? henyey_greenstein_phase(dot(dir, prev_dir_in), setup.g_seed)
        : mixed_phase(dot(dir, prev_dir_in), start_optics);
    float w0 = (q_seed > 1e-30f) ? phase_seed * INV_4PI / q_seed : 0.0f;

    // Seed polarization: unpolarized (the exact treatment would Mueller-
    // rotate the omega->view seed scatter; multiply-scattered light is
    // weakly polarized and the I-error is sub-percent - same approximation
    // as the CPU chain).
    float4 stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);

    float3 pos = start_pos;
    float3 current_dir = dir;
    float3 prev_dir = sun_dir; // direction before current propagation segment
    // NOTE: no start_optics.ssa factor - the host-side integrator's
    // beta_scat at the seed point already carries it (double-count removed,
    // mirroring the CPU fix).
    float weight = w0;

    KahanAccum total_I, total_Q, total_U, total_V;

    // Deep-twilight guiding parameters (port of the CPU chains): the
    // Dwivedi MIS ramps with the chain's SZA; below the 0.02 activation
    // the pure-phase path consumes identical RNG draws, so shallow streams
    // keep their structure. The field-forced (majorant-combined) channel
    // is active exactly when the kernel enabled forced mode under a field
    // (majorants guaranteed packed then; see SecondarySetup.use_forced).
    float d_frac = dwivedi_frac(setup.sza_deg);
    float d_beta = dwivedi_beta(setup.sza_deg);
    bool use_field_maj = field_present && (setup.use_forced != 0u);

    for (uint scatter_iter = 0; scatter_iter < setup.max_bounces; scatter_iter++) {
        // --- Decide scatter mode for this bounce ---
        // Fused scout + VSPG (port of the CPU chains): one shell walk
        // collects tau_max AND the altitude/SZA-importance segments that
        // place forced collisions where the sun still shines.
        bool forced_this_bounce = false;
        float tau_max = 0.0f;
        VspgSegment vspg_segs[VSPG_GPU_MAX_SEGMENTS];
        uint n_vspg_segs = 0;

        if (setup.use_forced != 0u) {
            ScoutResult scout = scout_with_vspg_segments(
                atm, fld, use_field_maj, pos, current_dir, wl_idx,
                setup.sza_deg, vspg_segs, n_vspg_segs);
            tau_max = scout.tau;
            // Force scatter only when path exits to space, optical depth is
            // within the useful range, and tau >= forced_tau_min. Without the
            // lower bound, chains at high altitude (tau ~ 1e-5) get killed by
            // weight *= (1 - exp(-tau)) ~ tau, losing 5 orders of magnitude
            // per bounce. The CPU falls back to analog mode for small tau.
            forced_this_bounce = !scout.hit_ground
                              && tau_max >= setup.forced_tau_min
                              && tau_max < FORCED_TAU_CUTOFF;
        }

        uint scatter_shell = 0;
        // Gray cloud channel: a cloud collision is a distinct vertex type
        // (pure depolarizing HG scatter, no SSA, no weight change). When set,
        // g_cloud_here carries the local delta-scaled asymmetry to the shared
        // NEE / direction-sampling block below.
        bool cloud_collision = false;
        float g_cloud_here = 0.0f;

        if (forced_this_bounce) {
            // Upfront forced scattering (unbiased): no analog walk, no
            // double-counting. tau_max is the (majorant-)COMBINED optical
            // depth (gas + gray 1D deck, or gas + per-shell field
            // majorant), so the weight is the (majorant-)combined
            // collision probability and the flight attenuates through AND
            // can collide in the cloud channel.
            float exp_neg_tau = exp(-tau_max);
            weight *= (1.0f - exp_neg_tau);
            if (weight < 1e-30f) break;
            // VSPG: sample the collision location from the pre-collected
            // importance segments (weight-corrected, unbiased; replaces
            // the plain truncated-exponential draw).
            VspgSample vs = vspg_sample_from_segments(vspg_segs, n_vspg_segs, tau_max, rng);
            float tau_s = vs.tau_s;
            weight *= vs.weight;
            AdvanceResult adv = advance_to_optical_depth(
                atm, fld, use_field_maj, pos, current_dir, tau_s, wl_idx);
            pos = adv.pos;
            current_dir = adv.dir;
            scatter_shell = adv.shell_idx;

            if (use_field_maj) {
                // FIELD: classify the majorant collision (real cloud /
                // real gas / null); nulls re-draw within the remaining
                // truncated budget. Port of the CPU field_forced_classify
                // (derivation and telescoping proof at the CPU scalar
                // chain's use_forced block). The classification uniform is
                // drawn ONLY in shells with a positive cloud majorant
                // (a majorant-clear shell is real gas with probability 1).
                float consumed = tau_s;
                uint fshell = adv.shell_idx;
                bool resolved = false;
                for (uint ev = 0; ev < FIELD_NULL_EVENT_LIMIT; ev++) {
                    float c_maj = field_shell_majorant(fld, fshell);
                    if (c_maj <= 0.0f) { resolved = true; break; } // real gas
                    float sigma_gas_m = atm[ATM_OPTICS_START
                        + (fshell * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];
                    float sigma_m = sigma_gas_m + c_maj;
                    float sigma_c_here = field_sigma_at(fld, pos);
                    float xi_cls = xorshift_f32(rng) * sigma_m;
                    if (xi_cls < sigma_c_here) {
                        cloud_collision = true;
                        g_cloud_here = field_g_at(fld, pos);
                        resolved = true;
                        break;
                    }
                    if (xi_cls < sigma_c_here + sigma_gas_m) { resolved = true; break; }
                    // NULL: continue the truncated flight in the remaining
                    // budget. Kill on an fp-exhausted budget: the correct
                    // continuation would carry weight * (1 - e^{-t_rem});
                    // the CPU threshold is 1e-12 (f64), below f32
                    // resolution near tau_max, so the f32 port kills at
                    // 1e-6 (expectation loss <= weight * 1e-6, far below
                    // the f32 MC noise floor -- the same acceptance class
                    // as FORCED_TAU_CUTOFF, where f32 rounds 1 - e^{-20}
                    // to exactly 1).
                    float t_rem = tau_max - consumed;
                    if (t_rem <= 1e-6f) break; // killed below
                    float e_rem = exp(-t_rem);
                    weight *= (1.0f - e_rem);
                    float xi2 = xorshift_f32(rng);
                    float d_tau = -log(1.0f - xi2 * (1.0f - e_rem) + 1e-30f);
                    AdvanceResult nadv = advance_to_optical_depth(
                        atm, fld, use_field_maj, pos, current_dir, d_tau, wl_idx);
                    pos = nadv.pos;
                    current_dir = nadv.dir;
                    fshell = nadv.shell_idx;
                    consumed += d_tau;
                }
                scatter_shell = fshell;
                if (!resolved) {
                    // Backstop kill (fp-exhausted budget or the null-event
                    // limit): terminate the particle with weight zero.
                    weight = 0.0f;
                    break;
                }
            } else {
                // 1D deck: vertex type from the exact extinction
                // conditional at the collision shell: cloud with
                // p = sigma_c / (sigma_c + sigma_gas) (the same
                // first-arrival law the analog race realizes). The type
                // probabilities cancel the per-type extinction factors
                // exactly, so no weight correction; a gas vertex carries
                // the gas SSA below, a cloud vertex is pure scattering.
                // The draw is taken ONLY when the collision shell carries
                // cloud, so clear-sky RNG streams keep their structure
                // (CPU ac673c7).
                float sigma_c_f = read_cloud_extinction(atm, scatter_shell);
                if (sigma_c_f > 0.0f) {
                    float sigma_gas_f = atm[ATM_OPTICS_START
                        + (scatter_shell * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE];
                    if (xorshift_f32(rng) < sigma_c_f / (sigma_c_f + sigma_gas_f)) {
                        cloud_collision = true;
                        // The gray deck's delta-scaled asymmetry.
                        g_cloud_here = atm[ATM_CLOUD_G_SCALED];
                    }
                }
            }
        } else {
            // Analog scatter WITH the gray cloud channel (decomposition
            // tracking; port of the CPU trace_secondary_chain analog arm,
            // ONE implementation for clear sky and both cloud
            // representations). The gas channel keeps the exponential
            // transform; a separate gray cloud Poisson process races over
            // each segment, the shorter free flight wins. The cloud
            // collision distance comes from exact inversion: the field DDA
            // (field_advance_to_tau) when a field is bound, or the analytic
            // per-shell inversion for the 1D shell deck (the CPU
            // cloud_flight_segment None arm). A cloud collision is pure HG
            // scatter (no SSA, no weight change); Beer-Lambert cloud on
            // every NEE leg; NO T_diff anywhere on chain paths. The cloud
            // budget is drawn ONCE per free flight from the SAME rng stream
            // (matching the CPU rng.tau per-flight order; in clear sky the
            // race is inert but the draw still advances the stream, like
            // the CPU), carried (undiminished by gas events) across shell
            // crossings, redrawn only after a collision or ground bounce.
            bool scatter_found = false;
            uint found_shell = 0u;
            // tau_c_remaining = -ln(1 - u + 1e-30), one draw at flight start.
            float tau_c_remaining = -log(1.0f - xorshift_f32(rng) + 1e-30f);

            for (uint step = 0; step < 200; step++) {
                float r = length(pos);
                int sidx = shell_index_binary(atm, r);
                if (sidx < 0) break;

                uint us = uint(sidx);
                ShellGeom sh = read_shell(atm, us);
                ShellOptics op = read_optics(atm, us, wl_idx);

                ShellBoundary bnd = next_shell_boundary(pos, current_dir, sh.r_inner, sh.r_outer);
                if (!bnd.found) break;

                // Gas free path (exponential transform on the gas channel);
                // clear air just walks to the boundary (free_path = INF, no
                // gas tau draw, matching the CPU's sigma < 1e-20 branch).
                float sigma = op.extinction;
                float cos_bias = 0.0f;
                float sigma_prime = sigma;
                float free_path = INFINITY;
                if (sigma >= 1e-20f) {
                    cos_bias = dot(current_dir, setup.term_axis_dir);
                    sigma_prime = sigma * (1.0f - setup.alpha_et * cos_bias);
                    if (sigma_prime <= 0.0f) sigma_prime = sigma;
                    float xi = xorshift_f32(rng);
                    free_path = -log(1.0f - xi + 1e-30f) / sigma_prime;
                }

                // Race the gray cloud channel over the segment up to the gas
                // event (gas scatter at free_path or boundary crossing).
                float gas_cap = min(free_path, bnd.dist);
                float cloud_dist = -1.0f;
                float tau_pass = 0.0f;
                if (field_present) {
                    cloud_dist = field_advance_to_tau(fld, pos, current_dir, gas_cap, tau_c_remaining);
                } else {
                    float sigma_c = read_cloud_extinction(atm, us);
                    if (sigma_c > 0.0f) {
                        float dist_c = tau_c_remaining / sigma_c;
                        if (dist_c <= gas_cap) cloud_dist = dist_c;
                        else tau_pass = sigma_c * gas_cap;
                    }
                }
                if (cloud_dist >= 0.0f) {
                    // Cloud wins. ET gas weight correction for the distance
                    // actually travelled (gray cloud ratio = 1).
                    if (setup.alpha_et > 0.0f && sigma >= 1e-20f) {
                        float et_arg = -setup.alpha_et * sigma * cos_bias * cloud_dist;
                        if (fabs(et_arg) < 80.0f) weight *= exp(et_arg);
                        else { weight = 0.0f; }
                    }
                    if (!isfinite(weight)) break;
                    pos = pos + current_dir * cloud_dist;
                    g_cloud_here = field_present ? field_g_at(fld, pos)
                                                 : atm[ATM_CLOUD_G_SCALED];
                    found_shell = us;
                    scatter_found = true;
                    cloud_collision = true;
                    break;
                } else {
                    // No cloud collision in this segment: consume its cloud tau.
                    if (field_present) {
                        tau_pass = field_tau_along(fld, pos, current_dir, gas_cap);
                    }
                    tau_c_remaining -= tau_pass;
                }

                if (free_path >= bnd.dist) {
                    if (setup.alpha_et > 0.0f && sigma >= 1e-20f) {
                        float et_arg = -setup.alpha_et * sigma * cos_bias * bnd.dist;
                        if (fabs(et_arg) < 80.0f) weight *= exp(et_arg);
                        else { weight = 0.0f; }
                    }
                    if (!isfinite(weight)) break;

                    float3 boundary_pos = pos + current_dir * bnd.dist;
                    boundary_pos = snap_to_radius(boundary_pos, bnd.is_outward ? sh.r_outer : sh.r_inner);

                    // Ground reflection.
                    if (!bnd.is_outward && length(boundary_pos) <= surface_radius + BOUNDARY_NUDGE_M) {
                        float3 normal = normalize(boundary_pos);
                        // Snap the bounce point ABOVE the surface (port of
                        // the CPU r_surface + 1e-3 snap, 2f09385; a 1 mm
                        // ledge vanishes in f32 at Earth radius, so use the
                        // boundary nudge, which exceeds the f32 ULP). The
                        // snapped point has a shell, so the ground shadow
                        // ray sees the real gas+cloud attenuation instead of
                        // escaping through an empty atmosphere.
                        float3 ground_pos = normal * (surface_radius + BOUNDARY_NUDGE_M);
                        float cos_sun_ground = dot(sun_dir, normal);
                        if (cos_sun_ground > 0.0f) {
                            float t_sun_gb = shadow_ray_transmittance_chain(
                                atm, fld, field_present, setup.cloud_channel != 0u,
                                ground_pos, sun_dir, wl_idx);
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
                        pos = ground_pos;
                        stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
                        // New free flight: redraw the cloud budget.
                        tau_c_remaining = -log(1.0f - xorshift_f32(rng) + 1e-30f);
                        continue;
                    }

                    // Refract and cross into the next shell; the cloud budget
                    // carries over undiminished by the crossing.
                    float n_from = read_refractive_index(atm, us);
                    uint next_s = bnd.is_outward ? us + 1 : us - 1;
                    float n_to = (next_s < atm_num_shells(atm)) ? read_refractive_index(atm, next_s) : 1.0f;
                    current_dir = refract_at_boundary(current_dir, boundary_pos, n_from, n_to);
                    pos = radial_nudge(boundary_pos, bnd.is_outward);
                    continue;
                }

                // Gas scatter within this shell.
                if (setup.alpha_et > 0.0f && sigma >= 1e-20f) {
                    float et_arg = -setup.alpha_et * sigma * cos_bias * free_path;
                    if (fabs(et_arg) < 80.0f) weight *= (sigma / sigma_prime) * exp(et_arg);
                    else { weight = 0.0f; }
                }
                if (!isfinite(weight)) break;
                pos = pos + current_dir * free_path;
                found_shell = us;
                scatter_found = true;
                break;
            }

            if (!scatter_found) break;
            scatter_shell = found_shell;
        }

        ShellOptics op = read_optics(atm, scatter_shell, wl_idx);

        // SSA: a cloud collision is pure scattering (absorption folded out of
        // the field), so no SSA factor; a gas collision carries the gas
        // single-scattering albedo as before.
        if (!cloud_collision) {
            weight *= op.ssa;
        }

        // NEE. A cloud vertex is a depolarizing HG (phase on I, output
        // unpolarized); a gas vertex applies the Mueller matrix to the
        // photon's actual Stokes state.
        if (isfinite(weight) && fabs(weight) > 1e-30f) {
            float t_sun_sec = shadow_ray_transmittance_chain(
                atm, fld, field_present, setup.cloud_channel != 0u, pos, sun_dir, wl_idx);
            if (t_sun_sec > 1e-30f) {
                float cos_angle_nee = clamp(dot(sun_dir, current_dir), -1.0f, 1.0f);
                float4 nee_stokes;
                if (cloud_collision) {
                    float p = henyey_greenstein_phase(cos_angle_nee, g_cloud_here);
                    nee_stokes = float4(stokes.x * p, 0.0f, 0.0f, 0.0f);
                } else {
                    float A_nee, B_nee, C_nee;
                    stokes_ABC(cos_angle_nee, op, A_nee, B_nee, C_nee);
                    float cos2phi_nee, sin2phi_nee;
                    scattering_plane_rotation(prev_dir, current_dir, -sun_dir, cos2phi_nee, sin2phi_nee);
                    if (!isfinite(cos2phi_nee)) { cos2phi_nee = 1.0f; sin2phi_nee = 0.0f; }
                    nee_stokes = scatter_stokes(A_nee, B_nee, C_nee, cos2phi_nee, sin2phi_nee, stokes);
                }

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

        // Sample the new direction. A cloud vertex scatters from the gray HG
        // lobe (2 dir draws: cos-theta, phi) and resets polarization; a gas
        // vertex samples the Rayleigh/HG mixture and updates Stokes.
        if (cloud_collision) {
            float ct_cloud = sample_henyey_greenstein(xorshift_f32(rng), g_cloud_here);
            float phi_cloud = 2.0f * PI * xorshift_f32(rng);
            float3 d_cloud = scatter_direction(current_dir, ct_cloud, phi_cloud);
            if (!isfinite(d_cloud.x) || (length(d_cloud) < 1e-10f)) break;
            prev_dir = current_dir;
            current_dir = d_cloud;
            stokes = float4(1.0f, 0.0f, 0.0f, 0.0f);
            continue;
        }

        // Sample new direction (gas vertex): Dwivedi/phase MIS mixture at
        // deep twilight (port of the CPU chains' MIS block, identical
        // arithmetic; the balance-heuristic weight corrects the SCALAR
        // intensity, and the Stokes update below uses the actual sampled
        // angle, so the polarization treatment is unchanged). Below the
        // 0.02 activation: the pure-phase path, draw-for-draw identical
        // to history.
        float cos_theta;
        float3 new_dir;
        bool mis_active = d_frac >= 0.02f;
        if (mis_active) {
            float3 local_up_here = normalize(pos);
            float alpha_p_mis = 1.0f - d_frac;
            float xi_branch = xorshift_f32(rng);
            if (xi_branch < d_frac) {
                // Dwivedi branch: horizontal-biased escape sampling in the
                // local (up, east, north) frame.
                float xi1 = xorshift_f32(rng);
                float xi2 = xorshift_f32(rng);
                float xi_sign = xorshift_f32(rng);
                DwivediSample dw = dwivedi_sample(xi1, xi2, xi_sign, d_beta);
                float sin_z = sqrt(max(1.0f - dw.cos_z * dw.cos_z, 0.0f));
                float3 arbitrary = (fabs(local_up_here.y) < 0.9f)
                    ? float3(0.0f, 1.0f, 0.0f)
                    : float3(1.0f, 0.0f, 0.0f);
                float3 east = normalize(cross(local_up_here, arbitrary));
                float3 north = cross(local_up_here, east);
                float3 d = normalize(local_up_here * dw.cos_z
                                     + east * (sin_z * cos(dw.phi))
                                     + north * (sin_z * sin(dw.phi)));
                cos_theta = clamp(dot(current_dir, d), -1.0f, 1.0f);
                float p_phase = mixed_phase(cos_theta, op) * INV_4PI;
                float p_dw = dwivedi_pdf(dw.cos_z, d_beta);
                float mis_denom = alpha_p_mis * p_phase + d_frac * p_dw;
                if (mis_denom > 1e-30f) {
                    weight *= p_phase / mis_denom;
                }
                new_dir = d;
            } else {
                // Phase branch (within MIS).
                cos_theta = clamp(
                    (xorshift_f32(rng) < op.rayleigh_fraction)
                        ? sample_rayleigh_analytic(xorshift_f32(rng))
                        : sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry),
                    -1.0f, 1.0f);
                float phi = 2.0f * PI * xorshift_f32(rng);
                new_dir = scatter_direction(current_dir, cos_theta, phi);
                float p_phase = mixed_phase(cos_theta, op) * INV_4PI;
                float cos_z_dw = dot(new_dir, local_up_here);
                float p_dw = dwivedi_pdf(cos_z_dw, d_beta);
                float mis_denom = alpha_p_mis * p_phase + d_frac * p_dw;
                if (mis_denom > 1e-30f) {
                    weight *= p_phase / mis_denom;
                }
            }
        } else {
            // Pure phase function: no Dwivedi, no MIS overhead.
            cos_theta = clamp(
                (xorshift_f32(rng) < op.rayleigh_fraction)
                    ? sample_rayleigh_analytic(xorshift_f32(rng))
                    : sample_henyey_greenstein(xorshift_f32(rng), op.asymmetry),
                -1.0f, 1.0f);
            float phi = 2.0f * PI * xorshift_f32(rng);
            new_dir = scatter_direction(current_dir, cos_theta, phi);
        }
        // Guard: if the sampled direction is zero/NaN, bail
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
// The v1 step-parallel hybrid_scatter kernel and its hybrid_los_prefix helper
// were REMOVED: no host pipeline state referenced them, and the v1 chain path
// embodied the retired T_diff eye-path estimator (chain code must carry NO
// T_diff; the gray cloud channel is explicit). hybrid_scatter_v2 below is the
// only hybrid kernel.
// ============================================================================

// ============================================================================
// Kernel 3a: hybrid_context_prefix
//
// One thread per (wavelength, step): precomputes the deterministic eye-path
// context for hybrid_scatter_v2 (see the HCTX constants). Runs ONCE per
// hybrid_scatter call; every ray chunk then reads the context instead of
// recomputing the prefix and substep DDAs. Fully parallel (no redundant
// uniform execution), so the context costs one pass regardless of how many
// watchdog-sized chunks the ray budget needs.
// ============================================================================

kernel void hybrid_context_prefix(
    device const float* atm    [[buffer(0)]],
    device const float* params [[buffer(1)]],
    device float*       ctx    [[buffer(2)]],
    device const float* fld    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (!atm_header_valid(atm)) return; // main kernel raises the sentinel
    uint num_wl = atm_num_wavelengths(atm);
    uint wl_idx = tid / HYBRID_LOS_STEPS;
    uint step_idx = tid % HYBRID_LOS_STEPS;
    if (wl_idx >= num_wl) return;

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    uint secondary_rays = read_secondary_rays(params);
    bool field_present  = read_field_present(params);

    float toa_radius = atm_toa_radius(atm);
    float surface_radius = atm_surface_radius(atm);
    // Eye-path entry snap (d4f682e class): an observer whose f32 radius
    // rounds below the surface makes the nearest LOS substeps land at
    // shell_index < 0 and silently vanish (review round 2).
    if (length(observer_pos) < surface_radius + BOUNDARY_NUDGE_M) {
        observer_pos = normalize(observer_pos) * (surface_radius + BOUNDARY_NUDGE_M);
    }

    // Same step geometry as hybrid_scatter_v2.
    bool valid = true;
    float ds = 0.0f;
    float step_start_s = 0.0f;
    {
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
                if (step_idx >= num_steps) {
                    valid = false;
                } else {
                    ds = los_end / float(num_steps);
                    step_start_s = float(step_idx) * ds;
                }
            }
        }
    }

    uint base = tid * HCTX_STRIDE;
    if (!valid) {
        ctx[base + HCTX_K_SUB] = as_type<float>(0u); // marks the step invalid
        return;
    }

    uint global_total_rays = read_photons_per_wl(params);
    if (global_total_rays == 0u) global_total_rays = secondary_rays;

    bool cloud_channel = false;
    uint k_sub = 1;
    float sub_ds = 0.0f;
    float sub_tau_cloud[CLOUD_MAX_SUBSTEPS];
    uint sub_ray_start[CLOUD_MAX_SUBSTEPS];
    uint sub_ray_count[CLOUD_MAX_SUBSTEPS];
    float tau_obs_prefix = 0.0f;
    float tau_cloud_prefix = 0.0f;

    if (valid) {
        cloud_channel = atm_has_cloud_channel(atm, field_present);
        float3 step_start = observer_pos + view_dir * step_start_s;
        // Exact cloud tau of the COARSE step: one field DDA or the 1D
        // analytic per-shell path lengths (CPU eye_step_cloud_tau).
        float tau_cloud_coarse = cloud_channel
            ? eye_step_cloud_tau(atm, fld, field_present, step_start, view_dir, ds)
            : 0.0f;
        if (tau_cloud_coarse > CLOUD_SUBSTEP_TAU) {
            k_sub = clamp(uint(ceil(tau_cloud_coarse / CLOUD_SUBSTEP_TAU)),
                          2u, CLOUD_MAX_SUBSTEPS);
        }
        sub_ds = ds / float(k_sub);

        if (k_sub == 1u) {
            sub_tau_cloud[0] = tau_cloud_coarse;
            sub_ray_start[0] = 0u;
            sub_ray_count[0] = global_total_rays;
        } else {
            // Importance weights: estimated contribution of each substep
            // (source strength times eye transmittance into it), the CPU
            // formula (tc + tg) * exp(-(cloud prefix + tc/2)).
            float sub_w[CLOUD_MAX_SUBSTEPS];
            float sum_w = 0.0f;
            float tau_pref = 0.0f;
            for (uint j = 0; j < k_sub; j++) {
                float3 sub_start = observer_pos
                    + view_dir * (step_start_s + float(j) * sub_ds);
                float tc = eye_step_cloud_tau(atm, fld, field_present,
                                              sub_start, view_dir, sub_ds);
                sub_tau_cloud[j] = tc;
                float3 mid = sub_start + view_dir * (sub_ds * 0.5f);
                float tg = 0.0f;
                int smid = shell_index_binary(atm, length(mid));
                if (smid >= 0) {
                    tg = atm[ATM_OPTICS_START
                        + (uint(smid) * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE]
                        * sub_ds;
                }
                float w = (tc + tg) * exp(-(tau_pref + 0.5f * tc));
                sub_w[j] = w;
                sum_w += w;
                tau_pref += tc;
            }
            // Allocation: n_j = round(N * w_j / sum_w), min 1 (the CPU
            // formula), cumulatively capped so the substep ranges PARTITION
            // [0, N) (the CPU can overshoot N by a few tail chains and
            // simply runs them; the GPU global ray domain is fixed by the
            // dispatch, so the tail is capped instead -- a variance-only
            // difference on near-zero-weight substeps).
            if (global_total_rays < k_sub) {
                // Fewer rays than substeps: a disjoint partition would
                // leave (k_sub - N) substeps with ray ranges beyond the
                // dispatch domain, silently DROPPING their in-scatter
                // contribution (an order-2+ bias, review round 2). Give
                // every substep the full range instead: each is averaged
                // over all N chains, which is unbiased (substeps become
                // correlated: variance only).
                for (uint j = 0; j < k_sub; j++) {
                    sub_ray_start[j] = 0u;
                    sub_ray_count[j] = global_total_rays;
                }
            } else {
            uint assigned = 0u;
            for (uint j = 0; j < k_sub; j++) {
                uint remaining = k_sub - 1u - j;
                uint nj;
                if (sum_w > 1e-30f) {
                    nj = uint(round(float(global_total_rays) * sub_w[j] / sum_w));
                } else {
                    nj = max(global_total_rays / k_sub, 1u);
                }
                nj = max(nj, 1u);
                uint cap = (global_total_rays > assigned + remaining)
                    ? (global_total_rays - assigned - remaining) : 1u;
                nj = min(nj, cap);
                sub_ray_start[j] = assigned;
                sub_ray_count[j] = nj;
                assigned += nj;
            }
            }
        }

        // Eye-path prefix optical depths to the START of this coarse step.
        // Gas: per-step midpoint quadrature over the previous coarse steps
        // (the CPU accumulates substep-resolved gas tau inside cloudy
        // prefix steps; gas extinction is smooth on the 8 km scale height,
        // so the coarse-midpoint prefix differs by far less than the f32
        // budget - a documented quadrature deviation, not an estimator
        // one). Cloud: ONE exact call over the whole prefix (integral
        // additivity makes it equal to the CPU per-substep accumulation).
        for (uint j = 0; j < step_idx; j++) {
            float sj = (float(j) + 0.5f) * ds;
            float3 pj = observer_pos + view_dir * sj;
            float rj = length(pj);
            if (rj <= toa_radius && rj >= surface_radius) {
                int sj_idx = shell_index_binary(atm, rj);
                if (sj_idx >= 0) {
                    tau_obs_prefix += atm[ATM_OPTICS_START
                        + (uint(sj_idx) * MAX_WAVELENGTHS + wl_idx) * ATM_OPTICS_STRIDE]
                        * ds;
                }
            }
        }
        if (cloud_channel && step_idx > 0u) {
            tau_cloud_prefix = eye_step_cloud_tau(atm, fld, field_present,
                                                  observer_pos, view_dir, step_start_s);
        }
    }

    ctx[base + HCTX_TAU_OBS] = tau_obs_prefix;
    ctx[base + HCTX_TAU_CLOUD] = tau_cloud_prefix;
    ctx[base + HCTX_K_SUB] = as_type<float>(k_sub);
    ctx[base + HCTX_SPARE] = 0.0f;
    for (uint j = 0; j < k_sub; j++) {
        ctx[base + HCTX_SUB_TAU + j] = sub_tau_cloud[j];
        ctx[base + HCTX_SUB_START + j] = as_type<float>(sub_ray_start[j]);
        ctx[base + HCTX_SUB_COUNT + j] = as_type<float>(sub_ray_count[j]);
    }
}

// ============================================================================
// Kernel 3b: hybrid_scatter_v2 (ray-parallel)
//
// Dispatch: (num_wavelengths, num_steps) threadgroups of 64 threads each.
//   tg_pos.x = wavelength index
//   tg_pos.y = COARSE LOS step index
//   thread_in_tg.x = chain lane within this step
//
// Port of the CPU hybrid_scatter_radiance (polarized per-wavelength path),
// including the cloud-adaptive eye-path substepping: each coarse step whose
// exact cloud tau exceeds CLOUD_SUBSTEP_TAU is subdivided (up to
// CLOUD_MAX_SUBSTEPS), the substeps get importance-allocated shares of the
// GLOBAL chain budget, and every substep carries its own order-1 NEE (gas +
// gray cloud), gas/cloud seed mixture, and beta_total chain scale.
// Cloud-free steps keep one substep and the pre-substepping structure.
//
// The step context (substep taus, budgets, prefix optical depths) is
// computed redundantly on ALL threads: the values are uniform, so the SIMD
// groups execute the block once in lockstep, and no threadgroup memory or
// barrier is needed for it (the per-substep arrays would not fit the old
// thread-0 + shared-scalar design).
//
// Split-dispatch contract (host: metal.rs hybrid_scatter): one dispatch
// covers global chain indices [ray_offset, ray_offset + secondary_rays).
// The substep budget allocation n_j partitions the GLOBAL budget
// [0, global_total_rays) identically in every chunk (it is deterministic),
// so each global chain index lands in exactly one substep with a
// well-defined local stratification index. Per-chain output is scaled by
// scale_m_j * (global_total_rays / n_j): the host divides the accumulated
// sum by global_total_rays, recovering the CPU's per-substep mean times
// scale_m_j. The deterministic order-1 terms are multiplied by THIS
// dispatch's ray count so they survive the same division across chunks.
//
// Output layout: output[wl * HYBRID_LOS_STEPS + step] -- host sums steps.
// Buffer 3 (fld) is the packed 3D cloud field (a stub when
// field_present == 0; never read then).
// ============================================================================

kernel void hybrid_scatter_v2(
    device const float* atm         [[buffer(0)]],
    device const float* params      [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    device const float* fld         [[buffer(3)]],
    device const float* ctx         [[buffer(4)]],
    uint3 tg_pos    [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_id    [[simdgroup_index_in_threadgroup]]
) {
    uint wl_idx = tg_pos.x;
    // Absolute LOS step index: threadgroup y plus the host's step-window
    // offset (0 when unsplit).
    uint step_idx = tg_pos.y + read_step_offset(params);
    uint ray_lane = tid_in_tg.x;

    // Uniform across the grid: all threads return before any barrier.
    if (!atm_header_valid(atm)) {
        if (wl_idx == 0 && step_idx == 0 && ray_lane == 0) output[0] = HEADER_SENTINEL;
        return;
    }
    uint num_wl = atm_num_wavelengths(atm);

    float3 observer_pos = read_observer(params);
    float3 view_dir     = read_view_dir(params);
    float3 sun_dir      = read_sun_dir(params);
    uint secondary_rays = read_secondary_rays(params);
    bool field_present  = read_field_present(params);

    float toa_radius = atm_toa_radius(atm);
    {
        // Eye-path entry snap: see hybrid_context_prefix (must match, or
        // the context and the chain walk disagree about the first substep).
        float srad = atm_surface_radius(atm);
        if (length(observer_pos) < srad + BOUNDARY_NUDGE_M) {
            observer_pos = normalize(observer_pos) * (srad + BOUNDARY_NUDGE_M);
        }
    }
    float surface_radius = atm_surface_radius(atm);

    // ── Uniform coarse-step geometry ─────────────────────────────────────
    bool valid = (wl_idx < num_wl);
    float ds = 0.0f;
    float step_start_s = 0.0f;
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
                if (step_idx >= num_steps) {
                    valid = false;
                } else {
                    ds = los_end / float(num_steps);
                    step_start_s = float(step_idx) * ds;
                }
            }
        }
    }

    // Global chain budget (stratification + substep allocation domain).
    uint global_total_rays = read_photons_per_wl(params);
    if (global_total_rays == 0u) global_total_rays = secondary_rays;
    // ray_offset from upper 32 bits of the seed (split-dispatch).
    ulong raw_seed = read_rng_seed(params);
    uint ray_offset = uint(raw_seed >> 32);
    ulong base_seed = raw_seed & 0xFFFFFFFFul;
    // Per-thread RNG stream, continued across this thread's substeps and
    // chains (unchanged formula: seeded by wavelength, coarse step, and
    // absolute thread lane).
    ulong seed_input = base_seed
        ^ (ulong(wl_idx) * 0x9E3779B97F4A7C15ul)
        ^ (ulong(step_idx) << 16)
        ^ (ulong(ray_lane + ray_offset) << 32);
    ulong rng = splitmix64(seed_input);

    // ── Precomputed eye-path context (hybrid_context_prefix) ────────────
    // Deterministic per geometry, identical in every chunk: read, not
    // recomputed (see the HCTX constants for why).
    bool cloud_channel = false;
    uint k_sub = 1;
    float sub_ds = 0.0f;
    float tau_obs_prefix = 0.0f;
    float tau_cloud_prefix = 0.0f;
    uint ctx_base = (wl_idx * HYBRID_LOS_STEPS + step_idx) * HCTX_STRIDE;
    if (valid) {
        cloud_channel = atm_has_cloud_channel(atm, field_present);
        k_sub = as_type<uint>(ctx[ctx_base + HCTX_K_SUB]);
        if (k_sub == 0u) {
            valid = false;
        } else {
            sub_ds = ds / float(k_sub);
            tau_obs_prefix = ctx[ctx_base + HCTX_TAU_OBS];
            tau_cloud_prefix = ctx[ctx_base + HCTX_TAU_CLOUD];
        }
    }

    float my_contribution = 0.0f;

    // ── Serial substep loop (uniform control flow; only the chain loop
    // inside diverges per thread, in trip count) ─────────────────────────
    if (valid) {
        float running_tau = tau_obs_prefix;
        float running_tau_cloud = tau_cloud_prefix;
        for (uint sub = 0; sub < k_sub; sub++) {
            float s = step_start_s + (float(sub) + 0.5f) * sub_ds;
            float3 scatter_pos = observer_pos + view_dir * s;
            float r = length(scatter_pos);
            if (r > toa_radius || r < surface_radius) continue;
            int my_sidx = shell_index_binary(atm, r);
            if (my_sidx < 0) continue;

            ShellOptics my_op = read_optics(atm, uint(my_sidx), wl_idx);
            float my_beta_scat = my_op.extinction * my_op.ssa;
            float tau_cloud_step = ctx[ctx_base + HCTX_SUB_TAU + sub];
            float beta_cloud = tau_cloud_step / sub_ds;

            // A substep contributes when EITHER channel scatters here (the
            // CPU skips only when both are absent).
            if (my_beta_scat < 1e-30f && beta_cloud <= 0.0f) {
                running_tau += my_op.extinction * sub_ds;
                running_tau_cloud += tau_cloud_step;
                continue;
            }

            // Chain-mode eye transmittance: Beer-Lambert for gas AND cloud
            // (explicit cloud scattering supplies the diffusion that T_diff
            // used to approximate; mixing them double-counts). Single exp
            // is more precise than multiplied exps in f32. Clear sky: the
            // cloud terms are identically zero.
            float t_obs = exp(-(running_tau + my_op.extinction * sub_ds * 0.5f
                                + running_tau_cloud + tau_cloud_step * 0.5f));
            if (t_obs < 1e-30f) break; // LOS opaque: later substeps darker still

            // Local asymmetry for this substep's cloud source terms
            // (order-1 NEE and cloud-seeded chains).
            float g_cloud_step = 0.0f;
            if (beta_cloud > 0.0f) {
                g_cloud_step = field_present ? field_g_at(fld, scatter_pos)
                                             : atm[ATM_CLOUD_G_SCALED];
            }

            // Order 1: deterministic NEE, computed ONCE per run (lane 0 of
            // the chunk containing global ray 0) and scaled by the GLOBAL
            // ray total: the host divides the accumulated sum by that same
            // total, restoring the deterministic term exactly. (The old
            // per-chunk recomputation scaled by each chunk's ray count
            // summed to the identical value but re-paid the substep shadow
            // DDAs in every one of the watchdog-sized field chunks.)
            if (ray_lane == 0u && ray_offset == 0u) {
                float t_sun = shadow_ray_transmittance_chain(
                    atm, fld, field_present, cloud_channel,
                    scatter_pos, sun_dir, wl_idx);
                if (t_sun > 1e-30f) {
                    float cos_theta_1 = dot(sun_dir, view_dir);
                    float A_1, B_1, C_1;
                    stokes_ABC(cos_theta_1, my_op, A_1, B_1, C_1);
                    float scale_1 = my_beta_scat * INV_4PI * t_sun * t_obs * sub_ds;
                    my_contribution += A_1 * scale_1 * float(global_total_rays);
                    // Cloud in-scatter source, order-1 NEE (gray channel):
                    // the deterministic sun -> cloud -> eye term,
                    // beta_cloud * P_HG(g*)/4pi * T_sun * T_obs * ds.
                    // Without it every path whose eye-nearest vertex is a
                    // cloud scatter is dropped (the dominant radiance under
                    // an overcast deck). The cloud vertex is a depolarizing
                    // HG: I-term only (the CPU order-1 cloud NEE, 0cc8bf5).
                    if (beta_cloud > 0.0f) {
                        float scale_c = beta_cloud
                            * henyey_greenstein_phase(cos_theta_1, g_cloud_step)
                            * INV_4PI * t_sun * t_obs * sub_ds;
                        my_contribution += scale_c * float(global_total_rays);
                    }
                }
            }

            // Orders 2+: MC chains. This substep owns global chain indices
            // [lo, lo + n_sub); this dispatch covers [ray_offset,
            // ray_offset + secondary_rays); this thread takes every 64th of
            // the intersection.
            uint n_sub = as_type<uint>(ctx[ctx_base + HCTX_SUB_COUNT + sub]);
            if (secondary_rays > 0u && n_sub > 0u) {
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
                setup.sza_deg = sza_deg;
                // Forced mode: on at deep twilight. The 1D deck composes
                // via the exact combined channel; a 3D field composes via
                // the majorant-combined channel + truncated null-collision
                // classification, which needs the v5 per-shell majorants.
                // A field packed without them (no uploaded atmosphere)
                // keeps the analog fallback.
                bool forced_ok = !field_present || field_has_shell_majorants(fld);
                setup.use_forced =
                    (sza_deg >= ZENITH_SZA_START_DEG && forced_ok) ? 1u : 0u;
                // The CPU forced_tau_min_for_sza sigmoid:
                // 0.05 - 0.03 * sigmoid(sza - 102).
                setup.forced_tau_min =
                    0.05f - 0.03f / (1.0f + exp(-(sza_deg - 102.0f)));
                setup.cloud_channel = cloud_channel ? 1u : 0u;
                setup.beta_seed = beta_cloud;
                setup.g_seed = g_cloud_step;
                setup.max_bounces = field_present ? HYBRID_FIELD_MAX_BOUNCES
                                                  : HYBRID_MAX_BOUNCES;

                // Cloud-seed mixture: each chain estimates
                // beta_gas*I_gas + beta_cloud*I_cloud with one type draw
                // whose selection probability cancels the per-type
                // coefficient, so the substep scale is beta_total (the CPU
                // scale_m). The (N / n_sub) factor folds this substep's
                // 1/n_sub mean against the host's global 1/N division.
                float scale_m = (my_beta_scat + beta_cloud) * t_obs * sub_ds
                    * (float(global_total_rays) / float(n_sub));

                uint lo = as_type<uint>(ctx[ctx_base + HCTX_SUB_START + sub]);
                uint hi = lo + n_sub;
                uint glo = max(lo, ray_offset);
                uint ghi = min(hi, ray_offset + secondary_rays);
                KahanAccum mc_I;
                if (glo < ghi) {
                    // First relative index >= (glo - ray_offset) owned by
                    // this lane (relative indices stripe mod 64 across the
                    // threadgroup, unchanged from the pre-substep kernel).
                    uint rel0 = glo - ray_offset;
                    uint first = rel0
                        + ((ray_lane + HYBRID_V2_THREADGROUP_SIZE
                            - (rel0 % HYBRID_V2_THREADGROUP_SIZE))
                           % HYBRID_V2_THREADGROUP_SIZE);
                    for (uint rel = first; rel + ray_offset < ghi;
                         rel += HYBRID_V2_THREADGROUP_SIZE) {
                        uint g = rel + ray_offset;
                        float4 chain = trace_secondary_chain(
                            atm, fld, field_present, scatter_pos, sun_dir,
                            wl_idx, my_op, view_dir, setup,
                            g - lo, n_sub, rng);
                        float val = chain.x * scale_m;
                        if (isfinite(val)) {
                            mc_I.add(val);
                        }
                    }
                }
                float mc_result = mc_I.result();
                if (isfinite(mc_result)) {
                    my_contribution += mc_result;
                }
            }

            running_tau += my_op.extinction * sub_ds;
            running_tau_cloud += tau_cloud_step;
        }
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
// Kernel 3f: field_tau_probe (G-DDA-PARITY)
//
// One thread per ray. Integrates the cloud optical depth along the ray via
// the device-side field DDA (field_tau_along) so the pure geometry can be
// compared bit-for-budget against the CPU tau_along. No Monte Carlo.
//
// rays buffer: one 8-f32 header (slot 0 = ray count as a bit pattern, rest
// pad), then 8 f32 per ray = vec4(p0.x, p0.y, p0.z, t_max), vec4(dir.xyz, _).
// The count bounds the thread: the dispatch grid rounds up to the
// threadgroup size, and unbounded excess threads read and write past the
// buffers (out-of-bounds on both sides for any non-multiple ray count).
// output buffer: tau[ray]
// ============================================================================

kernel void field_tau_probe(
    device const float* fld    [[buffer(0)]],
    device const float* rays   [[buffer(1)]],
    device float*       output [[buffer(2)]],
    uint                tid    [[thread_position_in_grid]]
) {
    uint n_rays = as_type<uint>(rays[0]);
    if (tid >= n_rays) return;
    // Header gate on the field buffer (mirrors the atmosphere gate).
    if (as_type<uint>(fld[FIELD_HDR_MAGIC]) != BUFFER_MAGIC
        || as_type<uint>(fld[FIELD_HDR_VERSION]) != BUFFER_VERSION) {
        if (tid == 0u) output[0] = HEADER_SENTINEL;
        return;
    }
    uint base = 8u + tid * 8u;
    float3 p0  = float3(rays[base + 0], rays[base + 1], rays[base + 2]);
    float  tmx = rays[base + 3];
    float3 dir = float3(rays[base + 4], rays[base + 5], rays[base + 6]);
    output[tid] = field_tau_along(fld, p0, dir, tmx);
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
