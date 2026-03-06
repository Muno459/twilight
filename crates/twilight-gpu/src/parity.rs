//! GPU-CPU parity coverage tracking and f32 reference implementations.
//!
//! This module provides:
//!
//! 1. A [`ParityFeature`] enum listing every CPU feature the GPU must implement
//! 2. A [`ParityCoverage`] struct tracking pass/fail/untested per feature per backend
//! 3. f32 reference implementations of key GPU algorithms that serve as the
//!    specification for what the GPU shaders must compute
//!
//! The f32 reference implementations are the "simulated GPU": they use f32
//! arithmetic to replicate what the GPU shader should do. When these match
//! the CPU f64 ground truth within f32 tolerance, the algorithm is correct
//! for GPU implementation.

use std::collections::HashMap;
use std::fmt;

use crate::buffers::{atm_offsets, PackedAtmosphere};
use crate::BackendKind;

// ── Parity feature enumeration ──────────────────────────────────────────

/// Categories of GPU features that must achieve parity with CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParityCategory {
    Geometry,
    Scattering,
    ShadowRay,
    HybridEngine,
    Precision,
    Buffer,
}

impl fmt::Display for ParityCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParityCategory::Geometry => write!(f, "Geometry"),
            ParityCategory::Scattering => write!(f, "Scattering"),
            ParityCategory::ShadowRay => write!(f, "Shadow Ray"),
            ParityCategory::HybridEngine => write!(f, "Hybrid Engine"),
            ParityCategory::Precision => write!(f, "Precision"),
            ParityCategory::Buffer => write!(f, "Buffer"),
        }
    }
}

/// Every CPU feature the GPU must implement correctly.
///
/// Each variant maps to a specific algorithm or property that the GPU shader
/// must replicate. The parity tests verify each one independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParityFeature {
    // -- Geometry --
    /// Ray-sphere intersection (discriminant, t_near, t_far)
    RaySphereIntersect,
    /// Shell index lookup from radius (binary search on GPU, linear on CPU)
    ShellLookup,
    /// Distance and direction to next shell boundary
    NextShellBoundary,

    // -- Scattering --
    /// Rayleigh phase function P(cos_theta)
    RayleighPhase,
    /// Henyey-Greenstein phase function P(cos_theta, g)
    HgPhase,
    /// Scatter direction from (incoming_dir, cos_theta, phi)
    ScatterDirection,
    /// Rayleigh angle sampling (CDF inversion)
    SampleRayleigh,
    /// HG angle sampling (CDF inversion)
    SampleHg,

    // -- Shadow Ray --
    /// Shell-by-shell trace (not analytical path)
    ShellByShellTrace,
    /// Snell's law refraction at shell boundaries
    SnellLawRefraction,
    /// Ground hit detection (returns T=0)
    GroundHitDetection,
    /// Early cutoff when tau > 50
    EarlyTauCutoff,

    // -- Hybrid Engine --
    /// LOS stepping with 200 quadrature points
    LosSteping,
    /// Next-event estimation at each LOS point
    Nee,
    /// Secondary MC chain tracing
    SecondaryChains,
    /// Importance sampling of scatter directions
    ImportanceSampling,
    /// Ground reflection with surface albedo
    GroundReflection,

    // -- Precision --
    /// Kahan compensated summation for f32 accumulation
    KahanSummation,
    /// RNG quality (xorshift64 state transitions, f32 conversion)
    RngQuality,

    // -- Buffer --
    /// Refractive index packing/unpacking roundtrip
    RefractiveIndexPacking,
    /// Buffer header magic/version validation
    HeaderValidation,
}

impl ParityFeature {
    /// Which category this feature belongs to.
    pub fn category(self) -> ParityCategory {
        match self {
            Self::RaySphereIntersect | Self::ShellLookup | Self::NextShellBoundary => {
                ParityCategory::Geometry
            }
            Self::RayleighPhase
            | Self::HgPhase
            | Self::ScatterDirection
            | Self::SampleRayleigh
            | Self::SampleHg => ParityCategory::Scattering,
            Self::ShellByShellTrace
            | Self::SnellLawRefraction
            | Self::GroundHitDetection
            | Self::EarlyTauCutoff => ParityCategory::ShadowRay,
            Self::LosSteping
            | Self::Nee
            | Self::SecondaryChains
            | Self::ImportanceSampling
            | Self::GroundReflection => ParityCategory::HybridEngine,
            Self::KahanSummation | Self::RngQuality => ParityCategory::Precision,
            Self::RefractiveIndexPacking | Self::HeaderValidation => ParityCategory::Buffer,
        }
    }

    /// Short display name for the feature.
    pub fn name(self) -> &'static str {
        match self {
            Self::RaySphereIntersect => "ray_sphere_intersect",
            Self::ShellLookup => "shell_lookup",
            Self::NextShellBoundary => "next_shell_boundary",
            Self::RayleighPhase => "rayleigh_phase",
            Self::HgPhase => "hg_phase",
            Self::ScatterDirection => "scatter_direction",
            Self::SampleRayleigh => "sample_rayleigh",
            Self::SampleHg => "sample_hg",
            Self::ShellByShellTrace => "shell_by_shell_trace",
            Self::SnellLawRefraction => "snell_law_refraction",
            Self::GroundHitDetection => "ground_hit_detection",
            Self::EarlyTauCutoff => "early_tau_cutoff",
            Self::LosSteping => "los_stepping",
            Self::Nee => "nee",
            Self::SecondaryChains => "secondary_chains",
            Self::ImportanceSampling => "importance_sampling",
            Self::GroundReflection => "ground_reflection",
            Self::KahanSummation => "kahan_summation",
            Self::RngQuality => "rng_quality",
            Self::RefractiveIndexPacking => "refractive_index_packing",
            Self::HeaderValidation => "header_validation",
        }
    }

    /// All features in category-grouped order.
    pub fn all() -> &'static [ParityFeature] {
        &[
            // Geometry
            Self::RaySphereIntersect,
            Self::ShellLookup,
            Self::NextShellBoundary,
            // Scattering
            Self::RayleighPhase,
            Self::HgPhase,
            Self::ScatterDirection,
            Self::SampleRayleigh,
            Self::SampleHg,
            // Shadow Ray
            Self::ShellByShellTrace,
            Self::SnellLawRefraction,
            Self::GroundHitDetection,
            Self::EarlyTauCutoff,
            // Hybrid Engine
            Self::LosSteping,
            Self::Nee,
            Self::SecondaryChains,
            Self::ImportanceSampling,
            Self::GroundReflection,
            // Precision
            Self::KahanSummation,
            Self::RngQuality,
            // Buffer
            Self::RefractiveIndexPacking,
            Self::HeaderValidation,
        ]
    }
}

impl fmt::Display for ParityFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── Parity status tracking ─────────────────────────────────────────────

/// Result of a single parity check.
#[derive(Debug, Clone)]
pub enum ParityStatus {
    /// Feature passed parity check.
    Pass,
    /// Feature failed parity check with reason.
    Fail(String),
    /// Feature has not been tested yet.
    Untested,
    /// Feature was skipped (e.g., no GPU hardware).
    Skipped(String),
}

impl fmt::Display for ParityStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail(reason) => write!(f, "FAIL: {}", reason),
            Self::Untested => write!(f, "-"),
            Self::Skipped(reason) => write!(f, "SKIP: {}", reason),
        }
    }
}

/// Tracks GPU-CPU parity coverage across all features and backends.
pub struct ParityCoverage {
    results: HashMap<(BackendKind, ParityFeature), ParityStatus>,
}

impl ParityCoverage {
    /// Create an empty coverage tracker with all features set to Untested.
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Record a parity result for a (backend, feature) pair.
    pub fn record(&mut self, backend: BackendKind, feature: ParityFeature, status: ParityStatus) {
        self.results.insert((backend, feature), status);
    }

    /// Get the status of a specific (backend, feature) pair.
    pub fn get(&self, backend: BackendKind, feature: ParityFeature) -> &ParityStatus {
        self.results
            .get(&(backend, feature))
            .unwrap_or(&ParityStatus::Untested)
    }

    /// Count results by status for a specific backend.
    pub fn summary(&self, backend: BackendKind) -> (usize, usize, usize, usize) {
        let total = ParityFeature::all().len();
        let mut pass = 0;
        let mut fail = 0;
        let mut skip = 0;
        for &feat in ParityFeature::all() {
            match self.get(backend, feat) {
                ParityStatus::Pass => pass += 1,
                ParityStatus::Fail(_) => fail += 1,
                ParityStatus::Skipped(_) => skip += 1,
                ParityStatus::Untested => {}
            }
        }
        let untested = total - pass - fail - skip;
        (pass, fail, skip, untested)
    }
}

/// Generate an ASCII parity coverage report.
///
/// Output is a formatted table showing which features pass/fail/untested
/// for each GPU backend. Intended for `--nocapture` test output.
pub fn parity_report(coverage: &ParityCoverage) -> String {
    let backends = [
        BackendKind::Metal,
        BackendKind::Vulkan,
        BackendKind::Cuda,
        BackendKind::Wgpu,
    ];

    let mut out = String::new();
    out.push_str("\nGPU-CPU Parity Coverage Report\n");
    out.push_str("================================================================\n");

    // Header row
    out.push_str(&format!(
        "{:<28} {:>6} {:>6} {:>6} {:>6}\n",
        "Feature", "Metal", "Vulkan", "CUDA", "wgpu"
    ));
    out.push_str("----------------------------------------------------------------\n");

    let mut current_category: Option<ParityCategory> = None;

    for &feat in ParityFeature::all() {
        let cat = feat.category();
        if current_category != Some(cat) {
            if current_category.is_some() {
                out.push('\n');
            }
            out.push_str(&format!("  {}\n", cat));
            current_category = Some(cat);
        }

        let mut row = format!("    {:<24}", feat.name());
        for &backend in &backends {
            let status = coverage.get(backend, feat);
            let cell = match status {
                ParityStatus::Pass => " PASS ",
                ParityStatus::Fail(_) => " FAIL ",
                ParityStatus::Untested => "   -  ",
                ParityStatus::Skipped(_) => " SKIP ",
            };
            row.push_str(cell);
        }
        out.push_str(&row);
        out.push('\n');
    }

    out.push_str("================================================================\n");

    // Summary per backend
    for &backend in &backends {
        let (pass, fail, skip, untested) = coverage.summary(backend);
        let total = ParityFeature::all().len();
        out.push_str(&format!(
            "  {}: {}/{} pass, {} fail, {} skip, {} untested\n",
            backend, pass, total, fail, skip, untested,
        ));
    }
    out.push('\n');

    out
}

// ══════════════════════════════════════════════════════════════════════
// f32 reference implementations (simulated GPU)
// ══════════════════════════════════════════════════════════════════════
//
// These functions use f32 arithmetic to simulate what the GPU shader
// should compute. They serve as the specification for the shader rewrite.

// ── f32 vector helpers ──────────────────────────────────────────────────

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn len3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = len3(v);
    if l < 1e-15 {
        return [0.0; 3];
    }
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn neg3(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

// ── Geometry ────────────────────────────────────────────────────────────

/// f32 ray-sphere intersection (GPU reference).
///
/// Returns `Some((t_near, t_far))` or `None` if the ray misses.
pub fn ray_sphere_intersect_f32(
    origin: [f32; 3],
    direction: [f32; 3],
    radius: f32,
) -> Option<(f32, f32)> {
    let a = dot3(direction, direction);
    let b = 2.0f32 * dot3(origin, direction);
    let c = dot3(origin, origin) - radius * radius;

    let disc = b * b - 4.0f32 * a * c;
    if disc < 0.0 {
        return None;
    }

    let sqrt_disc = disc.sqrt();
    let inv_2a = 0.5f32 / a;

    let t_near = (-b - sqrt_disc) * inv_2a;
    let t_far = (-b + sqrt_disc) * inv_2a;

    Some((t_near, t_far))
}

/// Binary search shell lookup from packed atmosphere buffer (GPU target).
///
/// O(log N) instead of the CPU's O(N) linear scan. This is the algorithm
/// the GPU shader must implement after the rewrite.
pub fn shell_index_binary_search(data: &[f32], num_shells: usize, radius: f32) -> Option<usize> {
    if num_shells == 0 {
        return None;
    }

    // Bounds check: below first shell or above last shell
    let r_inner_first = data[atm_offsets::SHELLS_START];
    let r_outer_last =
        data[atm_offsets::SHELLS_START + (num_shells - 1) * atm_offsets::SHELL_STRIDE + 1];

    if radius < r_inner_first || radius >= r_outer_last {
        return None;
    }

    // Binary search: find largest s such that r_inner[s] <= radius.
    // Since shells are contiguous (r_outer[s] == r_inner[s+1]),
    // this guarantees radius is in shell s.
    let mut lo: usize = 0;
    let mut hi: usize = num_shells;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let r_inner_mid = data[atm_offsets::SHELLS_START + mid * atm_offsets::SHELL_STRIDE];
        if r_inner_mid <= radius {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // lo is the first shell with r_inner > radius, so lo-1 has r_inner <= radius
    if lo == 0 {
        None
    } else {
        Some(lo - 1)
    }
}

/// f32 next shell boundary distance (GPU reference).
///
/// Returns `(distance, is_outward)` or `None`.
pub fn next_shell_boundary_f32(
    position: [f32; 3],
    direction: [f32; 3],
    r_inner: f32,
    r_outer: f32,
) -> Option<(f32, bool)> {
    // Try outer boundary first
    if let Some((t_near, t_far)) = ray_sphere_intersect_f32(position, direction, r_outer) {
        if t_near > 1e-5 {
            // Check if inner boundary is closer
            if let Some((inner_near, _)) = ray_sphere_intersect_f32(position, direction, r_inner) {
                if inner_near > 1e-5 && inner_near < t_near {
                    return Some((inner_near, false));
                }
            }
            return Some((t_near, true));
        }
        if t_far > 1e-5 {
            if let Some((inner_near, _)) = ray_sphere_intersect_f32(position, direction, r_inner) {
                if inner_near > 1e-5 && inner_near < t_far {
                    return Some((inner_near, false));
                }
            }
            return Some((t_far, true));
        }
    }

    // Fallback: inner only
    if let Some((inner_near, _)) = ray_sphere_intersect_f32(position, direction, r_inner) {
        if inner_near > 1e-5 {
            return Some((inner_near, false));
        }
    }

    None
}

// ── Scattering ──────────────────────────────────────────────────────────

/// f32 Rayleigh phase function (GPU reference).
pub fn rayleigh_phase_f32(cos_theta: f32) -> f32 {
    0.75f32 * (1.0f32 + cos_theta * cos_theta)
}

/// f32 Henyey-Greenstein phase function (GPU reference).
pub fn hg_phase_f32(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0f32 + g2 - 2.0f32 * g * cos_theta;
    (1.0f32 - g2) / (denom * denom.sqrt())
}

/// f32 scatter direction (GPU reference).
///
/// Given incoming direction, polar angle cos, and azimuth, compute
/// new direction using the same local frame construction as CPU.
pub fn scatter_direction_f32(dir: [f32; 3], cos_theta: f32, phi: f32) -> [f32; 3] {
    let sin_theta = ((1.0f32 - cos_theta * cos_theta).max(0.0f32)).sqrt();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    let w = dir;

    // Build local basis (same convention as CPU: use z-up unless parallel)
    let up = if w[2].abs() < 0.9f32 {
        [0.0f32, 0.0, 1.0]
    } else {
        [1.0f32, 0.0, 0.0]
    };

    let u = normalize3(cross3(w, up));
    let v = cross3(w, u);

    let result = [
        sin_theta * cos_phi * u[0] + sin_theta * sin_phi * v[0] + cos_theta * w[0],
        sin_theta * cos_phi * u[1] + sin_theta * sin_phi * v[1] + cos_theta * w[1],
        sin_theta * cos_phi * u[2] + sin_theta * sin_phi * v[2] + cos_theta * w[2],
    ];
    normalize3(result)
}

// ── Shadow Ray ──────────────────────────────────────────────────────────

/// f32 Snell's law refraction at a spherical boundary (GPU reference).
///
/// Returns the new direction after refraction (or reflection for TIR).
pub fn refract_at_boundary_f32(
    dir: [f32; 3],
    boundary_pos: [f32; 3],
    n_from: f32,
    n_to: f32,
) -> [f32; 3] {
    // Fast path: no refraction when indices match
    if (n_from - n_to).abs() < 1e-7 {
        return dir;
    }

    let outward = normalize3(boundary_pos);

    let cos_dir_normal = dot3(dir, outward);
    let normal = if cos_dir_normal < 0.0 {
        outward
    } else {
        neg3(outward)
    };

    let cos_i = -dot3(dir, normal);
    let eta = n_from / n_to;
    let k = 1.0f32 - eta * eta * (1.0f32 - cos_i * cos_i);

    if k < 0.0 {
        // Total internal reflection
        let reflected = add3(dir, scale3(normal, 2.0 * cos_i));
        return normalize3(reflected);
    }

    let cos_t = k.sqrt();
    let factor = eta * cos_i - cos_t;
    let refracted = add3(scale3(dir, eta), scale3(normal, factor));
    normalize3(refracted)
}

/// f32 shadow ray transmittance with refraction (GPU target).
///
/// This is the algorithm the GPU shader MUST implement after the rewrite.
/// It reads f32 data directly from the packed atmosphere buffer, uses
/// binary search for shell lookup, Snell's law at each boundary, and
/// Kahan summation for optical depth accumulation.
pub fn shadow_ray_transmittance_f32(
    packed: &PackedAtmosphere,
    scatter_pos: [f32; 3],
    sun_dir: [f32; 3],
    wavelength_idx: usize,
) -> f32 {
    let num_shells = packed.num_shells as usize;
    let data = &packed.data;

    let surface_radius = data[atm_offsets::SHELLS_START]; // r_inner of shell 0

    let mut pos = scatter_pos;
    let mut dir = sun_dir;

    // Kahan accumulator for optical depth
    let mut tau = 0.0f32;
    let mut tau_c = 0.0f32; // Kahan compensation

    for _ in 0..200 {
        let r = len3(pos);

        let shell_idx = match shell_index_binary_search(data, num_shells, r) {
            Some(idx) => idx,
            None => break,
        };

        let r_inner = data[atm_offsets::SHELLS_START + shell_idx * atm_offsets::SHELL_STRIDE];
        let r_outer = data[atm_offsets::SHELLS_START + shell_idx * atm_offsets::SHELL_STRIDE + 1];

        let optics_idx = shell_idx * 64 + wavelength_idx; // MAX_WAVELENGTHS=64
        let extinction = data[atm_offsets::OPTICS_START + optics_idx * atm_offsets::OPTICS_STRIDE];

        match next_shell_boundary_f32(pos, dir, r_inner, r_outer) {
            Some((dist, is_outward)) => {
                // Kahan sum: tau += extinction * dist
                let y = extinction * dist - tau_c;
                let t = tau + y;
                tau_c = (t - tau) - y;
                tau = t;

                // Refract at boundary
                let mut boundary_pos = add3(pos, scale3(dir, dist));
                // Snap to exact shell radius (matches GPU shader snap_to_radius)
                let target_r = if is_outward { r_outer } else { r_inner };
                let bp_r = len3(boundary_pos);
                if bp_r > 0.0 {
                    let s = target_r / bp_r;
                    boundary_pos = [
                        boundary_pos[0] * s,
                        boundary_pos[1] * s,
                        boundary_pos[2] * s,
                    ];
                }
                let n_from = data[atm_offsets::REFRACTIVE_INDEX_START + shell_idx];
                let next_shell = if is_outward {
                    shell_idx + 1
                } else {
                    shell_idx.wrapping_sub(1)
                };
                let n_to = if next_shell < num_shells {
                    data[atm_offsets::REFRACTIVE_INDEX_START + next_shell]
                } else {
                    1.0f32
                };

                dir = refract_at_boundary_f32(dir, boundary_pos, n_from, n_to);
                // Nudge past boundary RADIALLY, not along ray direction.
                // At Earth scale (r~6.4e6), f32 ULP is ~0.5m. For tangential
                // rays, a 1m nudge along the ray barely changes the radial
                // position and f32 rounds it back to the boundary -- the ray
                // gets stuck. Nudging radially by 2m guarantees the position
                // is clearly inside the next shell.
                let bp_r = len3(boundary_pos);
                let radial_dir = if bp_r > 1e-10 {
                    [
                        boundary_pos[0] / bp_r,
                        boundary_pos[1] / bp_r,
                        boundary_pos[2] / bp_r,
                    ]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let nudge_sign = if is_outward { 1.0f32 } else { -1.0f32 };
                pos = add3(boundary_pos, scale3(radial_dir, nudge_sign * 2.0));

                // Ground hit
                if !is_outward && len3(pos) <= surface_radius + 1.0 {
                    return 0.0;
                }
            }
            None => break,
        }

        if tau > 50.0 {
            return 0.0;
        }
    }

    (-tau).exp()
}

// ── Precision ───────────────────────────────────────────────────────────

/// Kahan compensated summation in f32 (GPU target).
///
/// Returns a more accurate sum than naive f32 accumulation, especially
/// when values span many orders of magnitude (common in deep twilight
/// where contributions range from 1e-30 to 1e-5).
pub fn kahan_sum_f32(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32;
    for &v in values {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Naive (non-compensated) f32 summation for comparison.
pub fn naive_sum_f32(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in values {
        sum += v;
    }
    sum
}

// ── RNG ─────────────────────────────────────────────────────────────────

/// xorshift64 state advance and f32 conversion (GPU reference).
///
/// The state transition is identical to CPU (u64 xorshift). The difference
/// is the final conversion: f32 has 24-bit mantissa, so only the top
/// 24 bits of the state contribute to the result. CPU uses 53 bits for f64.
///
/// Returns `(f32_value, f64_value)` so tests can compare both conversions
/// from the same state.
pub fn xorshift_advance(state: &mut u64) -> (f32, f64) {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;

    // CPU f64 conversion: 53-bit mantissa
    let f64_val = (x >> 11) as f64 / (1u64 << 53) as f64;

    // GPU f32 conversion: same state, cast to f32
    // The GPU shader does: float(x >> 11) / float(1 << 53)
    // But float(1<<53) is exact in f32, and float(x>>11) loses
    // bits beyond the 24-bit mantissa. Equivalent to:
    let f32_val = ((x >> 11) as f64 / (1u64 << 53) as f64) as f32;

    (f32_val, f64_val)
}

// ══════════════════════════════════════════════════════════════════════
// Unit tests for f32 reference implementations
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle;

    #[test]
    fn binary_search_matches_linear_on_oracle_atmosphere() {
        let atm = oracle::oracle_atmosphere();
        let packed = PackedAtmosphere::pack(&atm);
        let num_shells = packed.num_shells as usize;

        // Test 100 altitudes from 0 to 110 km
        for i in 0..=110 {
            let alt = i as f64 * 1000.0;
            let radius = twilight_core::atmosphere::EARTH_RADIUS_M + alt;

            let cpu_idx = atm.shell_index(radius);
            let gpu_idx = shell_index_binary_search(&packed.data, num_shells, radius as f32);

            assert_eq!(
                cpu_idx, gpu_idx,
                "Mismatch at altitude {}m: CPU={:?}, GPU binary={:?}",
                alt, cpu_idx, gpu_idx,
            );
        }
    }

    #[test]
    fn binary_search_boundary_cases() {
        let atm = oracle::oracle_atmosphere();
        let packed = PackedAtmosphere::pack(&atm);
        let ns = packed.num_shells as usize;

        // Below surface
        let r_below = (twilight_core::atmosphere::EARTH_RADIUS_M - 1.0) as f32;
        assert_eq!(shell_index_binary_search(&packed.data, ns, r_below), None);

        // Above TOA
        let r_above = (twilight_core::atmosphere::EARTH_RADIUS_M + 100_001.0) as f32;
        assert_eq!(shell_index_binary_search(&packed.data, ns, r_above), None);

        // At exact surface
        let r_surface = twilight_core::atmosphere::EARTH_RADIUS_M as f32;
        assert_eq!(
            shell_index_binary_search(&packed.data, ns, r_surface),
            Some(0),
        );
    }

    #[test]
    fn f32_rayleigh_matches_f64() {
        let cos_values = [-1.0, -0.7, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0];
        for &mu in &cos_values {
            let f64_val = twilight_core::scattering::rayleigh_phase(mu);
            let f32_val = rayleigh_phase_f32(mu as f32) as f64;
            let rel_err = (f64_val - f32_val).abs() / f64_val.abs().max(1e-30);
            assert!(
                rel_err < 1e-6,
                "Rayleigh phase mismatch at cos_theta={}: f64={}, f32={}, rel={}",
                mu,
                f64_val,
                f32_val,
                rel_err,
            );
        }
    }

    #[test]
    fn f32_hg_matches_f64() {
        let cos_values = [-1.0, -0.5, 0.0, 0.5, 0.9, 1.0];
        let g_values = [0.0, 0.3, 0.65, 0.85, -0.5];
        for &mu in &cos_values {
            for &g in &g_values {
                let f64_val = twilight_core::scattering::henyey_greenstein_phase(mu, g);
                let f32_val = hg_phase_f32(mu as f32, g as f32) as f64;
                let rel_err = (f64_val - f32_val).abs() / f64_val.abs().max(1e-30);
                assert!(
                    rel_err < 1e-4,
                    "HG phase mismatch at cos={}, g={}: f64={}, f32={}, rel={}",
                    mu,
                    g,
                    f64_val,
                    f32_val,
                    rel_err,
                );
            }
        }
    }

    #[test]
    fn kahan_sum_beats_naive_on_extreme_range() {
        // Values spanning 1e-25 to 1e-5: naive f32 loses the small values
        let mut values = Vec::new();
        for _ in 0..1000 {
            values.push(1e-25f32);
        }
        values.push(1e-5f32);
        values.push(-1e-5f32);
        for _ in 0..1000 {
            values.push(1e-25f32);
        }

        let f64_sum: f64 = values.iter().map(|&v| v as f64).sum();
        let kahan = kahan_sum_f32(&values) as f64;
        let naive = naive_sum_f32(&values) as f64;

        let kahan_err = (f64_sum - kahan).abs();
        let naive_err = (f64_sum - naive).abs();

        // Kahan should be at least as good as naive
        assert!(
            kahan_err <= naive_err + 1e-35,
            "Kahan ({:.6e}, err={:.6e}) should be >= naive ({:.6e}, err={:.6e})",
            kahan,
            kahan_err,
            naive,
            naive_err,
        );
    }

    #[test]
    fn xorshift_state_transitions_match_cpu() {
        let mut gpu_state = 42u64;
        let mut cpu_state = 42u64;

        for i in 0..100 {
            let (gpu_f32, _gpu_f64) = xorshift_advance(&mut gpu_state);
            let cpu_f64 = twilight_core::photon::xorshift_f64(&mut cpu_state);

            // States must be identical (same u64 xorshift algorithm)
            assert_eq!(
                gpu_state, cpu_state,
                "State diverged at step {}: gpu={}, cpu={}",
                i, gpu_state, cpu_state,
            );

            // f32 value should be close to f64 (within f32 precision)
            let diff = (gpu_f32 as f64 - cpu_f64).abs();
            assert!(
                diff < 1e-7,
                "RNG value mismatch at step {}: f32={}, f64={}, diff={}",
                i,
                gpu_f32,
                cpu_f64,
                diff,
            );
        }
    }

    #[test]
    fn ray_sphere_f32_matches_f64_for_earth_scale() {
        use twilight_core::atmosphere::EARTH_RADIUS_M;
        use twilight_core::geometry::{ray_sphere_intersect as rs_f64, Vec3};

        let r_toa = (EARTH_RADIUS_M + 100_000.0) as f32;

        let origin = [EARTH_RADIUS_M as f32, 0.0f32, 0.0f32];
        let dir = [1.0f32, 0.0, 0.0];

        let f32_result = ray_sphere_intersect_f32(origin, dir, r_toa);
        let f64_result = rs_f64(
            Vec3::new(EARTH_RADIUS_M, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            EARTH_RADIUS_M + 100_000.0,
        );

        assert!(f32_result.is_some());
        assert!(f64_result.is_some());

        let (t_near_f32, t_far_f32) = f32_result.unwrap();
        let hit_f64 = f64_result.unwrap();

        // t_far should be ~100km; f32 at this scale has ULP ~8m
        let t_far_err = (t_far_f32 as f64 - hit_f64.t_far).abs();
        assert!(
            t_far_err < 50.0,
            "t_far: f32={}, f64={}, err={}m",
            t_far_f32,
            hit_f64.t_far,
            t_far_err,
        );

        let t_near_err = (t_near_f32 as f64 - hit_f64.t_near).abs();
        assert!(
            t_near_err < 50.0,
            "t_near: f32={}, f64={}, err={}m",
            t_near_f32,
            hit_f64.t_near,
            t_near_err,
        );
    }

    #[test]
    fn scatter_direction_f32_produces_unit_vector() {
        let dirs: [[f32; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            normalize3([0.3, 0.7, -0.5]),
        ];
        let cos_thetas = [-0.9f32, -0.3, 0.0, 0.5, 0.95];
        let phis = [0.0f32, 1.0, 3.14, 5.5];

        for &dir in &dirs {
            for &ct in &cos_thetas {
                for &phi in &phis {
                    let result = scatter_direction_f32(dir, ct, phi);
                    let len = len3(result);
                    assert!(
                        (len - 1.0).abs() < 1e-5,
                        "scatter_direction not unit: dir={:?}, ct={}, phi={}, |result|={}",
                        dir,
                        ct,
                        phi,
                        len,
                    );
                }
            }
        }
    }

    #[test]
    fn refract_f32_identity_when_equal_n() {
        let dir = normalize3([0.3f32, -0.9, 0.1]);
        let boundary = [100.0f32, 0.0, 0.0];
        let result = refract_at_boundary_f32(dir, boundary, 1.000293, 1.000293);
        for i in 0..3 {
            assert!(
                (result[i] - dir[i]).abs() < 1e-6,
                "Identity refraction failed: dir={:?}, result={:?}",
                dir,
                result,
            );
        }
    }

    #[test]
    fn coverage_tracker_basic_operations() {
        let mut cov = ParityCoverage::new();

        // Initially everything is untested
        let (pass, fail, skip, untested) = cov.summary(BackendKind::Metal);
        assert_eq!(pass, 0);
        assert_eq!(fail, 0);
        assert_eq!(skip, 0);
        assert_eq!(untested, ParityFeature::all().len());

        // Record some results
        cov.record(
            BackendKind::Metal,
            ParityFeature::RayleighPhase,
            ParityStatus::Pass,
        );
        cov.record(
            BackendKind::Metal,
            ParityFeature::ShellLookup,
            ParityStatus::Fail("linear not binary".to_string()),
        );
        cov.record(
            BackendKind::Metal,
            ParityFeature::KahanSummation,
            ParityStatus::Skipped("not implemented yet".to_string()),
        );

        let (pass, fail, skip, _) = cov.summary(BackendKind::Metal);
        assert_eq!(pass, 1);
        assert_eq!(fail, 1);
        assert_eq!(skip, 1);

        // Vulkan should still be all untested
        let (pass, _, _, untested) = cov.summary(BackendKind::Vulkan);
        assert_eq!(pass, 0);
        assert_eq!(untested, ParityFeature::all().len());
    }

    #[test]
    fn parity_report_is_nonempty() {
        let cov = ParityCoverage::new();
        let report = parity_report(&cov);
        assert!(!report.is_empty());
        assert!(report.contains("Parity Coverage Report"));
        assert!(report.contains("Metal"));
        assert!(report.contains("Vulkan"));
    }

    #[test]
    fn feature_all_has_expected_count() {
        // 3 + 5 + 4 + 5 + 2 + 2 = 21
        assert_eq!(ParityFeature::all().len(), 21);
    }

    #[test]
    fn feature_categories_are_correct() {
        assert_eq!(
            ParityFeature::RaySphereIntersect.category(),
            ParityCategory::Geometry,
        );
        assert_eq!(
            ParityFeature::RayleighPhase.category(),
            ParityCategory::Scattering,
        );
        assert_eq!(
            ParityFeature::ShellByShellTrace.category(),
            ParityCategory::ShadowRay,
        );
        assert_eq!(
            ParityFeature::LosSteping.category(),
            ParityCategory::HybridEngine,
        );
        assert_eq!(
            ParityFeature::KahanSummation.category(),
            ParityCategory::Precision,
        );
        assert_eq!(
            ParityFeature::HeaderValidation.category(),
            ParityCategory::Buffer,
        );
    }
}
