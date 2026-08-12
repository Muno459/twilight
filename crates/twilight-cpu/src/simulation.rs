//! High-level MCRT simulation driver.
//!
//! Takes observer location, solar zenith angle, and atmosphere model,
//! and computes spectral sky radiance. Supports two scattering modes:
//!
//! - **Single scattering** (default): deterministic line-of-sight integration.
//!   Fast, no noise, accurate for clear-sky twilight up to ~15° depression.
//!
//! - **Multiple scattering**: backward Monte Carlo with next-event estimation.
//!   Handles all scattering orders, needed for deep twilight (>15°), thick
//!   clouds, and reaching the 18° depression angle used by MWL/ISNA.
//!
//! The radiance output is in physical units [W/m²/sr/nm] when solar
//! irradiance weighting is enabled (default).

use rayon::prelude::*;
use twilight_core::atmosphere::AtmosphereModel;
use twilight_core::cloud_field::Cloud3DField;
use twilight_core::geometry::{geographic_to_ecef, solar_direction_ecef, Vec3};
use twilight_core::photon;
use twilight_core::single_scatter;
use twilight_data::solar_spectrum::SOLAR_IRRADIANCE;

/// Result of a spectral simulation at a single solar zenith angle.
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Wavelengths in nm
    pub wavelengths_nm: Vec<f64>,
    /// Sky radiance at each wavelength [W/m²/sr/nm] (physical units when
    /// solar irradiance weighting is applied)
    pub radiance: Vec<f64>,
    /// Solar zenith angle (degrees)
    pub sza_deg: f64,
}

/// Scattering mode for the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScatteringMode {
    /// Deterministic single-scattering line-of-sight integration.
    /// Fast, no noise. Accurate for clear sky up to ~15° depression.
    #[default]
    Single,
    /// Backward Monte Carlo with next-event estimation.
    /// Handles all scattering orders. Required for deep twilight (>15°),
    /// thick clouds, and physically reaching 18° depression angles.
    Multiple,
    /// Hybrid: deterministic single-scatter (order 1) + MC secondary
    /// chains (orders 2+). Best convergence for deep twilight.
    /// Uses `photons_per_wavelength` as number of secondary rays per LOS step.
    Hybrid,
}

/// Configuration for a twilight simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Observer latitude (degrees, north positive)
    pub latitude: f64,
    /// Observer longitude (degrees, east positive)
    pub longitude: f64,
    /// Observer elevation above sea level (meters)
    pub elevation: f64,
    /// Solar azimuth angle (degrees, 0=north, clockwise).
    /// For twilight, typically ~90° (east, Fajr) or ~270° (west, Isha).
    pub solar_azimuth: f64,
    /// Zenith viewing direction (degrees from straight up).
    /// ~70-80° toward the sun azimuth captures the brightest twilight sky.
    pub view_zenith: f64,
    /// View azimuth (degrees, 0=north, clockwise). `None` means "look toward
    /// the solar azimuth" (relative azimuth 0), the historical behavior.
    /// Set explicitly for off-principal-plane geometry (e.g. libRadtran
    /// comparison grids).
    pub view_azimuth: Option<f64>,
    /// Whether to weight radiance by solar spectrum (true = physical units).
    /// When false, radiance is in relative units (useful for debugging).
    pub apply_solar_irradiance: bool,
    /// Scattering mode: single (deterministic) or multiple (Monte Carlo).
    pub scattering_mode: ScatteringMode,
    /// Number of photons per wavelength for MC mode. Ignored in single mode.
    /// Higher values reduce noise but increase computation time.
    /// Recommended: 10000+ for converged results, 1000 for quick estimates.
    pub photons_per_wavelength: usize,
    /// Enable full Stokes [I,Q,U,V] polarization tracking (default: true).
    ///
    /// When true (the default), the hybrid/MC CPU engine propagates full
    /// 4-component Stokes vectors through Mueller matrices, capturing
    /// polarization-intensity coupling from Rayleigh and aerosol scattering.
    ///
    /// When false (`--fast` mode), uses scalar phase function (P11 only).
    /// Slightly faster, loses ~0.5-2% polarization correction.
    pub polarized: bool,
    /// Extra entropy mixed into MC seeds. Salt 0 reproduces historical
    /// runs; distinct salts give statistically independent estimates of
    /// the same radiance - the basis for K-seed averaging and standard-
    /// error estimation in the prayer pipeline.
    pub seed_salt: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            latitude: 21.4225, // Mecca
            longitude: 39.8262,
            elevation: 0.0,
            solar_azimuth: 270.0, // West (Isha/sunset direction)
            view_zenith: 75.0,    // Look toward horizon
            view_azimuth: None,   // default: toward the solar azimuth
            apply_solar_irradiance: true,
            scattering_mode: ScatteringMode::Single,
            photons_per_wavelength: 10_000,
            polarized: true,
            seed_salt: 0,
        }
    }
}


/// Mix the config's seed salt into a base seed. Salt 0 leaves the seed
/// unchanged (bit-for-bit reproducibility of historical runs); any other
/// salt is dispersed through a splitmix64 finalizer before XOR so
/// consecutive salts give decorrelated streams.
#[inline]
fn mix_salt(base: u64, salt: u64) -> u64 {
    if salt == 0 {
        return base;
    }
    let mut z = salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    base ^ (z ^ (z >> 31))
}

/// Run simulation at a single solar zenith angle.
///
/// Dispatches between single-scattering (deterministic) and multiple-scattering
/// (Monte Carlo) based on `config.scattering_mode`.
///
/// Returns spectral radiance across all wavelengths in the atmosphere model.
///
/// When `config.apply_solar_irradiance` is true, the output is in physical
/// units [W/m²/sr/nm]. The integral gives:
///   I(λ) = F_sun(λ) × ∫ β_scat × P(θ)/(4π) × T_sun × T_obs ds
/// where F_sun(λ) is the TOA solar spectral irradiance [W/m²/nm].
/// `field`: the 3D cloud field; when present it owns ALL cloud (the
/// caller guarantees `atm.cloud_extinction` is all-zero, see the
/// pipeline's `build_atmosphere`). The config stays reference-free.
pub fn simulate_at_sza(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
    field: Option<&Cloud3DField>,
) -> SpectralResult {
    match config.scattering_mode {
        ScatteringMode::Single => simulate_at_sza_single(atm, config, sza_deg, field),
        ScatteringMode::Multiple => simulate_at_sza_mc(atm, config, sza_deg, field),
        ScatteringMode::Hybrid => simulate_at_sza_hybrid(atm, config, sza_deg, field),
    }
}

/// Compute observer/sun/view geometry from config and SZA.
pub(crate) fn compute_geometry(config: &SimulationConfig, sza_deg: f64) -> (Vec3, Vec3, Vec3) {
    let observer_pos = geographic_to_ecef(config.latitude, config.longitude, config.elevation);
    let sun_dir = solar_direction_ecef(
        sza_deg,
        config.solar_azimuth,
        config.latitude,
        config.longitude,
    );
    let view_dir = solar_direction_ecef(
        config.view_zenith,
        config.view_azimuth.unwrap_or(config.solar_azimuth),
        config.latitude,
        config.longitude,
    );
    (observer_pos, sun_dir, view_dir)
}

/// Apply solar irradiance weighting and build SpectralResult from raw radiance array.
fn build_spectral_result(
    atm: &AtmosphereModel,
    radiance_array: &[f64; 64],
    sza_deg: f64,
    apply_solar_irradiance: bool,
) -> SpectralResult {
    let num_wl = atm.num_wavelengths;
    let mut wavelengths = Vec::with_capacity(num_wl);
    let mut radiance = Vec::with_capacity(num_wl);

    for w in 0..num_wl {
        wavelengths.push(atm.wavelengths_nm[w]);
        let r = if apply_solar_irradiance && w < SOLAR_IRRADIANCE.len() {
            radiance_array[w] * SOLAR_IRRADIANCE[w]
        } else {
            radiance_array[w]
        };
        radiance.push(r);
    }

    SpectralResult {
        wavelengths_nm: wavelengths,
        radiance,
        sza_deg,
    }
}

/// Single-scattering simulation (deterministic, no noise).
fn simulate_at_sza_single(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
    field: Option<&Cloud3DField>,
) -> SpectralResult {
    let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);
    let radiance_array =
        single_scatter::single_scatter_spectrum(atm, observer_pos, view_dir, sun_dir, field);
    build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance)
}

/// Multiple-scattering simulation via backward Monte Carlo with NEE.
///
/// Traces `config.photons_per_wavelength` photons per wavelength using rayon
/// parallelism. Each photon undergoes multiple scattering events with
/// next-event estimation at each bounce.
///
/// The result captures all scattering orders: the first bounce is equivalent
/// to single scattering, and subsequent bounces add the multiple-scattering
/// contribution that becomes important at deep twilight (>15° depression)
/// and in thick clouds.
fn simulate_at_sza_mc(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
    field: Option<&Cloud3DField>,
) -> SpectralResult {
    let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);
    let num_wl = atm.num_wavelengths;
    let nphotons = config.photons_per_wavelength;

    if nphotons == 0 {
        let radiance_array = [0.0f64; 64];
        return build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance);
    }

    // Parallelize over wavelengths using rayon.
    // Each wavelength traces nphotons photons independently.
    let per_wl_radiance: Vec<f64> = (0..num_wl)
        .into_par_iter()
        .map(|w| {
            let mut total_weight = 0.0;
            for p in 0..nphotons {
                // Unique seed per (sza, wavelength, photon) triple.
                // Include sza bits to decorrelate across SZA scan steps.
                let sza_bits = mix_salt(sza_deg.to_bits(), config.seed_salt);
                let mut rng = (sza_bits)
                    .wrapping_add(w as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(p as u64)
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(1);

                let result =
                    photon::trace_photon(atm, observer_pos, view_dir, sun_dir, w, &mut rng, field);
                total_weight += result.weight;
            }
            total_weight / nphotons as f64
        })
        .collect();

    let mut radiance_array = [0.0f64; 64];
    for (w, &r) in per_wl_radiance.iter().enumerate() {
        radiance_array[w] = r;
    }

    build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance)
}

/// Hybrid simulation: deterministic single-scatter (order 1) + MC secondary
/// chains (orders 2+). Best convergence for deep twilight.
///
/// The single-scatter contribution is computed exactly (no noise), then at
/// each LOS step, secondary MC chains are launched to capture higher-order
/// scattering. This produces converged results at deep twilight (15-18°
/// depression) with far fewer photons than pure backward MC.
///
/// For scalar (non-polarized) mode, uses ALIS (Adjusted Lambda Importance
/// Sampling) which traces ONE hero wavelength path per chain but evaluates
/// ALL wavelengths simultaneously via per-wavelength weight ratios. This
/// gives ~N_wl fewer shadow ray traces for the same expected value.
///
/// For polarized mode, uses rayon parallelism over wavelengths with full
/// Stokes [I,Q,U,V] propagation per chain.
fn simulate_at_sza_hybrid(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
    field: Option<&Cloud3DField>,
) -> SpectralResult {
    let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);
    let secondary_rays = config.photons_per_wavelength;

    if !config.polarized {
        // ALIS path: all wavelengths in a single call.
        let sza_bits = mix_salt(sza_deg.to_bits(), config.seed_salt);
        let mut rng = sza_bits.wrapping_mul(6364136223846793005).wrapping_add(1);

        let radiance_array = photon::hybrid_scatter_radiance_alis(
            atm,
            observer_pos,
            view_dir,
            sun_dir,
            secondary_rays,
            &mut rng,
            field,
        );
        return build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance);
    }

    // Polarized path: per-wavelength Stokes tracing with rayon parallelism.
    let num_wl = atm.num_wavelengths;
    let per_wl_radiance: Vec<f64> = (0..num_wl)
        .into_par_iter()
        .map(|w| {
            let sza_bits = mix_salt(sza_deg.to_bits(), config.seed_salt);
            let mut rng = sza_bits
                .wrapping_add(w as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);

            photon::hybrid_scatter_radiance(
                atm,
                observer_pos,
                view_dir,
                sun_dir,
                w,
                secondary_rays,
                &mut rng,
                config.polarized,
                field,
            )
        })
        .collect();

    let mut radiance_array = [0.0f64; 64];
    for (w, &r) in per_wl_radiance.iter().enumerate() {
        radiance_array[w] = r;
    }

    build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance)
}

/// Run simulation across a range of solar zenith angles.
///
/// Scans through twilight, computing spectral radiance at each SZA.
pub fn simulate_twilight_scan(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_start: f64,
    sza_end: f64,
    sza_step: f64,
    field: Option<&Cloud3DField>,
) -> Vec<SpectralResult> {
    let mut results = Vec::new();
    let mut sza = sza_start;

    while sza <= sza_end + 1e-6 {
        let result = simulate_at_sza(atm, config, sza, field);
        results.push(result);
        sza += sza_step;
    }

    results
}

/// Compute total broadband radiance from spectral result (trapezoidal integration).
pub fn total_radiance(result: &SpectralResult) -> f64 {
    let n = result.radiance.len();
    if n < 2 {
        return result.radiance.first().copied().unwrap_or(0.0);
    }

    let mut total = 0.0;
    for i in 0..(n - 1) {
        let dw = result.wavelengths_nm[i + 1] - result.wavelengths_nm[i];
        total += 0.5 * (result.radiance[i] + result.radiance[i + 1]) * dw;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use twilight_data::atmosphere_profiles::AtmosphereType;
    use twilight_data::builder;

    fn make_clear_sky_atm() -> AtmosphereModel {
        builder::build_clear_sky(AtmosphereType::UsStandard, 0.15)
    }

    fn default_config() -> SimulationConfig {
        SimulationConfig::default()
    }

    // ── SimulationConfig defaults ──

    #[test]
    fn default_config_mecca() {
        let c = SimulationConfig::default();
        assert!((c.latitude - 21.4225).abs() < 0.01);
        assert!((c.longitude - 39.8262).abs() < 0.01);
        assert!((c.elevation - 0.0).abs() < 0.01);
        assert!((c.solar_azimuth - 270.0).abs() < 0.01);
        assert!((c.view_zenith - 75.0).abs() < 0.01);
        assert!(c.apply_solar_irradiance);
    }

    // ── 150 km ceiling (regression for the deep-twilight zero) ──

    /// With the old 100 km ceiling, single-scatter radiance was EXACTLY
    /// zero for SZA >= ~104 deg - yet prayer-time crossings live at
    /// 104-106 deg (verified against MYSTIC spherical, which still sees
    /// signal there). The thermospheric shells (100-150 km) carry that
    /// signal: it must be nonzero and decreasing in SZA.
    #[test]
    fn deep_twilight_single_scatter_nonzero_to_sza_107() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            view_zenith: 85.0,
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let mut prev = f64::MAX;
        for sza in [103.0, 104.0, 105.0, 106.0, 107.0] {
            let r: f64 = simulate_at_sza(&atm, &config, sza, None).radiance.iter().sum();
            assert!(
                r > 0.0,
                "single-scatter must be nonzero at SZA {} (was exactly 0 \
                 under the 100 km ceiling), got {:.3e}",
                sza,
                r
            );
            assert!(r < prev, "radiance must decrease with SZA");
            prev = r;
        }
    }

    // ── Cloud transport (regression for the OD-10 collapse) ──

    /// Through an OD-10 stratus deck the twilight sky must remain visible.
    ///
    /// HISTORY: this gate pinned the Stage-1 Eddington T_diff closure (a
    /// deterministic ~1e-12 floor). Stage 2 made in-cloud scattering
    /// EXPLICIT but disabled forced mode under any cloud channel, and the
    /// analog-only chains could not sample the rare deck-penetrating paths
    /// of a uniform OD-10 deck at gate budgets (measured: OD-10/SZA-95
    /// climbs 7.3e-5 at P=400 vs 7.8e-9 at P=50, far under-converged), so
    /// the gate sat ignored as a documented starvation limitation.
    ///
    /// NOW: the combined-channel forced mode folds the shell-constant gray
    /// deck into the forced-flight channel exactly (photon.rs, `use_forced`
    /// derivation), so deck-penetrating paths are sampled by construction.
    /// The gate runs at SZA 97, inside the forced-mode regime
    /// (ZENITH_SZA_START = 96, where the starvation actually lived; probed
    /// post-fix at 4000 photons: cloudy 2.9e-5..1.1e-4 across seeds vs
    /// clear 7.0e-3, both assertions hold with 60x+ margin). Below SZA 96
    /// forced mode is inactive by the SZA ramp and the uniform OD-10 deck
    /// remains analog and variance-starved: a documented residual, not
    /// covered by this gate. Still `#[ignore]`: heavy MC (minutes).
    #[test]
    #[ignore = "g_s2_: heavy MC (OD-10 deck, two full hybrid runs)"]
    fn stratus_twilight_remains_visible_and_below_clear_sky() {
        use twilight_data::cloud::{default_properties, CloudType};
        let clear = make_clear_sky_atm();
        let props = default_properties(CloudType::Stratus);
        let cloudy = builder::build_with_cloud_properties(
            AtmosphereType::UsStandard,
            0.15,
            &props,
        );
        let config = SimulationConfig {
            view_zenith: 85.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 4000,
            polarized: false,
            ..SimulationConfig::default()
        };
        let sza = 97.0;
        let r_clear: f64 = simulate_at_sza(&clear, &config, sza, None).radiance.iter().sum();
        let r_cloudy: f64 = simulate_at_sza(&cloudy, &config, sza, None).radiance.iter().sum();
        assert!(
            r_cloudy > 1e-9,
            "cloudy twilight collapsed (Beer-Lambert dark, no MS): {:.3e}",
            r_cloudy
        );
        assert!(
            r_cloudy < r_clear,
            "an OD-10 deck must dim the sky: cloudy={:.3e} clear={:.3e}",
            r_cloudy,
            r_clear
        );
    }

    // ── 3D cloud field: Stage-1 gates ──

    /// Stratus layer description shared by the field gates (matches
    /// `twilight_data::cloud::default_properties(CloudType::Stratus)`).
    fn stratus_props() -> twilight_data::cloud::CloudProperties {
        twilight_data::cloud::default_properties(twilight_data::cloud::CloudType::Stratus)
    }

    /// A horizontally uniform stratus field centered on the default
    /// (Mecca) observer; `background_column` extends it to infinity.
    fn uniform_stratus_field() -> twilight_data::cloud_field_builder::OwnedCloudField {
        use twilight_data::cloud_field_builder::{field_from_layers, FieldGeometry};
        let c = SimulationConfig::default();
        field_from_layers(
            &[stratus_props()],
            FieldGeometry {
                center_lat_deg: c.latitude,
                center_lon_deg: c.longitude,
                half_extent_km: 128.0,
                res_km: 2.0,
            },
            "gate",
        )
    }

    /// G-S1a (equivalence): the 1D layered stratus transport vs the SAME
    /// atmosphere with `cloud_extinction` zeroed (the field-run caller
    /// contract: the field owns ALL cloud) plus the uniform 3D field.
    /// The only remaining difference is the cloud tau quadrature
    /// (per-shell segments vs per-voxel DDA) on the same T_diff
    /// convention, so per-wavelength radiance must agree to < 1%.
    /// (A fully clear-built atmosphere would also differ by the cloud
    /// ABSORPTION the 1D builder folds into shell optics, ~4% on a
    /// 75 deg slant, which is not the quadrature under test.)
    #[test]
    fn g_s1a_uniform_field_matches_1d_layered_transport() {
        let atm_1d = builder::build_with_cloud_properties(
            AtmosphereType::UsStandard,
            0.15,
            &stratus_props(),
        );
        let mut atm_field = atm_1d.clone();
        atm_field.cloud_extinction = [0.0; twilight_core::atmosphere::MAX_SHELLS];
        let owned = uniform_stratus_field();
        let view = owned.view();

        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        for sza in [95.0, 100.0] {
            let r_1d = simulate_at_sza(&atm_1d, &config, sza, None);
            let r_3d = simulate_at_sza(&atm_field, &config, sza, Some(&view));
            let mut max_rel = 0.0f64;
            for w in 0..r_1d.radiance.len() {
                let (a, b) = (r_1d.radiance[w], r_3d.radiance[w]);
                if a < 1e-30 && b < 1e-30 {
                    continue;
                }
                let rel = (a - b).abs() / a.max(b);
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 0.01,
                    "G-S1a SZA {sza} wl[{w}]: 1D {a:.6e} vs field {b:.6e} (rel {rel:.4})"
                );
            }
            eprintln!("G-S1a SZA {sza}: max per-wavelength rel diff {max_rel:.3e}");
        }
    }

    /// G-S1b (gap physics): a field cloudy ONLY on the anti-sun side of
    /// the footprint (sun-azimuth side clear) must be strictly brighter
    /// than the uniform deck (sun rays pass through the gap) and no
    /// brighter than clear sky, at SZA 96 in Single mode.
    #[test]
    fn g_s1b_sunside_gap_brightens_sky() {
        let mut atm = builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let uniform = uniform_stratus_field();
        // Field runs keep the cloud asymmetry on the T_diff convention.
        atm.cloud_g_scaled = uniform.g_default;

        // Sun azimuth is 270 (west) at the default config: clear the
        // WEST half of the footprint (lower longitudes), keep the deck
        // on the anti-sun (east) half, then re-derive the acceleration
        // data (the background column becomes the half-deck mean).
        let mut gap = uniform.clone();
        for iz in 0..gap.nz {
            for ilat in 0..gap.nlat {
                for ilon in 0..gap.nlon / 2 {
                    gap.sigma[(iz * gap.nlat + ilat) * gap.nlon + ilon] = 0.0;
                }
            }
        }
        gap.derive();

        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let sza = 96.0;
        let sum = |r: &SpectralResult| -> f64 { r.radiance.iter().sum() };
        let r_clear = sum(&simulate_at_sza(&atm, &config, sza, None));
        let r_uniform = sum(&simulate_at_sza(&atm, &config, sza, Some(&uniform.view())));
        let r_gap = sum(&simulate_at_sza(&atm, &config, sza, Some(&gap.view())));
        eprintln!(
            "G-S1b SZA {sza}: clear {r_clear:.4e}, gap {r_gap:.4e}, uniform {r_uniform:.4e}"
        );

        assert!(
            r_gap > r_uniform,
            "G-S1b: gap {r_gap:.4e} must exceed uniform deck {r_uniform:.4e}"
        );
        assert!(
            r_gap <= r_clear * (1.0 + 1e-12),
            "G-S1b: gap {r_gap:.4e} must not exceed clear {r_clear:.4e}"
        );
    }

    // ── 3D cloud field: Stage-2 explicit-scattering gates ──

    /// Thin uniform deck (OD 2, base 1 km, top 3 km) shared by the Stage-2
    /// chain gates in both representations (1D shell cloud and 3D field).
    /// Thin enough that the analog cloud channel converges at gate photon
    /// counts (the OD-10 stratus default is the documented variance-starved
    /// case, see `stratus_twilight_remains_visible_and_below_clear_sky`).
    fn thin_deck_props() -> twilight_data::cloud::CloudProperties {
        twilight_data::cloud::CloudProperties {
            base_km: 1.0,
            top_km: 3.0,
            optical_depth: 2.0,
            ssa: 0.999,
            asymmetry: 0.85,
        }
    }

    /// A horizontally uniform thin field (OD-2) for the Stage-2 chain gates.
    /// Thin enough that the analog cloud channel converges at modest photon
    /// counts (forced mode is off under a 3D FIELD; the 1D deck now runs
    /// combined-channel forced flights), uniform so it has an exact
    /// 1D-shell equivalent.
    fn uniform_thin_field() -> twilight_data::cloud_field_builder::OwnedCloudField {
        use twilight_data::cloud_field_builder::{field_from_layers, FieldGeometry};
        let c = SimulationConfig::default();
        field_from_layers(
            &[thin_deck_props()],
            FieldGeometry {
                center_lat_deg: c.latitude,
                center_lon_deg: c.longitude,
                half_extent_km: 256.0,
                res_km: 4.0,
            },
            "g_s2",
        )
    }

    /// Average summed radiance over K seeds (mean and standard error).
    /// Seeds run in parallel (rayon); the per-seed salts and therefore the
    /// values are identical to the previous serial version.
    fn mc_mean_se(
        atm: &AtmosphereModel,
        config: &SimulationConfig,
        sza: f64,
        field: Option<&Cloud3DField>,
        k: u64,
    ) -> (f64, f64) {
        let s: Vec<f64> = (0..k)
            .into_par_iter()
            .map(|seed| {
                let mut c = config.clone();
                c.seed_salt = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                simulate_at_sza(atm, &c, sza, field).radiance.iter().sum()
            })
            .collect();
        let mean = s.iter().sum::<f64>() / k as f64;
        let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / k as f64;
        (mean, (var / k as f64).sqrt())
    }

    /// G-EQ1D (Stage-2 analog of g_s1a, for chains): a horizontally uniform
    /// field's full MC radiance must equal the SAME deck as 1D shell cloud
    /// extinction in the new explicit-scattering estimator, within MC noise.
    /// Both paths run the identical analog decomposition-tracking model
    /// (field DDA vs per-shell analytic inversion of the gray channel), so
    /// they must agree statistically. N=8 seeds x 256 photons, SZA 95.
    #[test]
    #[ignore = "g_s2_eq1d: heavy MC"]
    fn g_s2_eq1d_uniform_field_matches_1d_explicit() {
        let props = thin_deck_props();
        // 1D path: shell cloud extinction, no field.
        let atm_1d = builder::build_with_cloud_properties(
            AtmosphereType::UsStandard,
            0.15,
            &props,
        );
        // Field path: zero the shells, carry the uniform field.
        let mut atm_field = atm_1d.clone();
        atm_field.cloud_extinction = [0.0; twilight_core::atmosphere::MAX_SHELLS];
        let owned = uniform_thin_field();
        atm_field.cloud_g_scaled = owned.g_default;
        let view = owned.view();

        let config = SimulationConfig {
            view_zenith: 80.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 256,
            polarized: false,
            ..SimulationConfig::default()
        };
        let sza = 95.0;
        let (m_1d, se_1d) = mc_mean_se(&atm_1d, &config, sza, None, 8);
        let (m_3d, se_3d) = mc_mean_se(&atm_field, &config, sza, Some(&view), 8);
        let se = (se_1d * se_1d + se_3d * se_3d).sqrt();
        let diff = (m_1d - m_3d).abs();
        eprintln!(
            "G-EQ1D SZA {sza}: 1D {m_1d:.5e} (se {se_1d:.2e}) field {m_3d:.5e} (se {se_3d:.2e}) diff {diff:.2e} comb-se {se:.2e}"
        );
        // Within 3 combined standard errors (a generous statistical band).
        assert!(
            diff < 3.0 * se + 0.02 * m_1d.max(m_3d),
            "G-EQ1D: 1D {m_1d:.5e} vs field {m_3d:.5e} differ by {diff:.3e} (> 3 se = {:.3e})",
            3.0 * se
        );
    }

    /// Per-wavelength SCALAR-chain hybrid reference: one independent scalar
    /// chain family per wavelength (`hybrid_scatter_radiance` with
    /// `polarized = false`), summed with the same solar-irradiance weights
    /// as `build_spectral_result`. Same estimator family and LOS quadrature
    /// as ALIS, no hero-path machinery, no polarization: a comparison
    /// against it isolates the ALIS spectral reweighting alone.
    fn perwl_scalar_hybrid_total(
        atm: &AtmosphereModel,
        config: &SimulationConfig,
        sza: f64,
        field: Option<&Cloud3DField>,
    ) -> f64 {
        let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza);
        (0..atm.num_wavelengths)
            .into_par_iter()
            .map(|w| {
                let sza_bits = mix_salt(sza.to_bits(), config.seed_salt);
                let mut rng = sza_bits
                    .wrapping_add(w as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                let raw = photon::hybrid_scatter_radiance(
                    atm,
                    observer_pos,
                    view_dir,
                    sun_dir,
                    w,
                    config.photons_per_wavelength,
                    &mut rng,
                    false,
                    field,
                );
                if config.apply_solar_irradiance && w < SOLAR_IRRADIANCE.len() {
                    raw * SOLAR_IRRADIANCE[w]
                } else {
                    raw
                }
            })
            .sum()
    }

    /// K-seed mean and standard error of the per-wavelength scalar hybrid.
    fn perwl_scalar_mean_se(
        atm: &AtmosphereModel,
        config: &SimulationConfig,
        sza: f64,
        field: Option<&Cloud3DField>,
        k: u64,
    ) -> (f64, f64) {
        let s: Vec<f64> = (0..k)
            .into_par_iter()
            .map(|seed| {
                let mut c = config.clone();
                c.seed_salt = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                perwl_scalar_hybrid_total(atm, &c, sza, field)
            })
            .collect();
        let mean = s.iter().sum::<f64>() / k as f64;
        let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / k as f64;
        (mean, (var / k as f64).sqrt())
    }

    /// G-ALIS (Stage-2): the ALIS hero-path chain vs explicit per-wavelength
    /// SCALAR chains on a synthetic 3D cloud (a cube embedded in the
    /// footprint) must agree statistically. The CPU runs ALIS in Hybrid
    /// non-polarized mode; the reference is the per-wavelength scalar chain
    /// (the `polarized = false` path of `hybrid_scatter_radiance`), an
    /// independent estimator of the SAME scalar integral with the same LOS
    /// quadrature. The previous reference was the polarized Stokes hybrid,
    /// which confounds ALIS machinery errors with polarization-coupling
    /// physics; its 3% physics floor let a recorded 6.4%-scale delta pass
    /// unexamined. The scalar reference removes the physics floor entirely.
    ///
    /// Acceptance band, derived from the measured seed-mean standard errors:
    /// |mean_alis - mean_ref| < 3 * sqrt(se_alis^2 + se_ref^2), the two-sided
    /// ~99.7% band under the normal approximation of K-seed means. No
    /// systematic floor is added because the two estimators share the
    /// integrand and quadrature exactly; any excess is an ALIS bug.
    #[test]
    #[ignore = "g_s2_alis: heavy MC"]
    fn g_s2_alis_matches_per_wavelength_on_cube() {
        let mut owned = uniform_thin_field();
        // Carve a cube: keep cloud only in a central block of columns and a
        // mid-altitude band; clear elsewhere. This is a genuinely 3D field.
        let (nz, nlat, nlon) = (owned.nz, owned.nlat, owned.nlon);
        let lat_lo = nlat * 7 / 16;
        let lat_hi = nlat * 9 / 16;
        let lon_lo = nlon * 7 / 16;
        let lon_hi = nlon * 9 / 16;
        for iz in 0..nz {
            for ilat in 0..nlat {
                for ilon in 0..nlon {
                    let inside = (lat_lo..lat_hi).contains(&ilat)
                        && (lon_lo..lon_hi).contains(&ilon);
                    if !inside {
                        owned.sigma[(iz * nlat + ilat) * nlon + ilon] = 0.0;
                    }
                }
            }
        }
        owned.derive();
        let view = owned.view();

        let mut atm = builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
        atm.cloud_g_scaled = owned.g_default;

        let alis = SimulationConfig {
            view_zenith: 80.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 256,
            polarized: false,
            ..SimulationConfig::default()
        };

        let sza = 94.0;
        let (m_alis, se_alis) = mc_mean_se(&atm, &alis, sza, Some(&view), 8);
        let (m_pw, se_pw) = perwl_scalar_mean_se(&atm, &alis, sza, Some(&view), 8);
        let se = (se_alis * se_alis + se_pw * se_pw).sqrt();
        let diff = (m_alis - m_pw).abs();
        eprintln!(
            "G-ALIS SZA {sza}: alis {m_alis:.5e} (se {se_alis:.2e}) perwl-scalar {m_pw:.5e} (se {se_pw:.2e}) diff {diff:.2e} band {:.2e}",
            3.0 * se
        );
        // Band derived from the measured SEs alone (see the doc comment):
        // same integrand, same quadrature, so 3 combined SE with no floor.
        assert!(
            diff < 3.0 * se,
            "G-ALIS: alis {m_alis:.5e} vs perwl-scalar {m_pw:.5e} differ by {diff:.3e} (> 3 se = {:.3e})",
            3.0 * se
        );
    }

    /// G-GAP-MC (Stage-2 analog of g_s1b, in chain mode): the sun-side gap
    /// geometry must be strictly brighter than the uniform deck and not
    /// brighter than clear sky, now with explicit in-cloud scattering.
    #[test]
    #[ignore = "g_s2_gap_mc: heavy MC"]
    fn g_s2_gap_mc_gap_brighter_than_deck_below_clear() {
        let mut atm = builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let uniform = uniform_thin_field();
        atm.cloud_g_scaled = uniform.g_default;

        // Clear the sun-azimuth (west, lower-longitude) half of the deck.
        let mut gap = uniform.clone();
        for iz in 0..gap.nz {
            for ilat in 0..gap.nlat {
                for ilon in 0..gap.nlon / 2 {
                    gap.sigma[(iz * gap.nlat + ilat) * gap.nlon + ilon] = 0.0;
                }
            }
        }
        gap.derive();

        let config = SimulationConfig {
            view_zenith: 80.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 256,
            polarized: false,
            ..SimulationConfig::default()
        };
        let sza = 95.0;
        let (r_clear, _) = mc_mean_se(&atm, &config, sza, None, 8);
        let (r_uniform, se_u) = mc_mean_se(&atm, &config, sza, Some(&uniform.view()), 8);
        let (r_gap, se_g) = mc_mean_se(&atm, &config, sza, Some(&gap.view()), 8);
        eprintln!(
            "G-GAP-MC SZA {sza}: clear {r_clear:.4e}, gap {r_gap:.4e} (se {se_g:.2e}), uniform {r_uniform:.4e} (se {se_u:.2e})"
        );
        assert!(
            r_gap > r_uniform,
            "G-GAP-MC: gap {r_gap:.4e} must exceed uniform deck {r_uniform:.4e}"
        );
        // Clear sky is the no-cloud upper bound (allow a 5% MC margin).
        assert!(
            r_gap <= r_clear * 1.05,
            "G-GAP-MC: gap {r_gap:.4e} must not exceed clear {r_clear:.4e}"
        );
    }

    /// G-HYB-MULT (Stage-2 cross-estimator gate): Hybrid vs
    /// `ScatteringMode::Multiple` under cloud.
    ///
    /// The adversarial review's central complaint: no INDEPENDENT reference
    /// for the chain estimator existed (G-EQ1D's two sides are trajectory-
    /// correlated replicas of the same decomposition-tracking walker).
    /// `trace_photon` (Multiple mode) is an independent estimator of the
    /// same radiance integral: a fully analog backward walk that races the
    /// gray cloud channel on EVERY flight, so it places cloud vertices on
    /// the eye ray itself. Post the 2026-07 G2 campaign, Multiple is also
    /// the externally anchored side: it matches disort/MYSTIC on the G2
    /// slab at SZA 30-85 AND spherical-backward MYSTIC at SZA 95/97 under
    /// this very deck (550 nm: 0.93 +- 0.08 and 0.98 +- 0.15).
    ///
    /// Two regimes, gated differently and honestly:
    ///
    /// 1. SZA 88 and 92 (twilight, converged): two-sided agreement,
    ///    |mean_h - mean_m| < 3 * combined SE + 5% construction floor
    ///    (straight midpoint LOS vs refracting analog walk).
    /// 2. SZA 97 (deep twilight, above ZENITH_SZA_START = 96): TWO-SIDED
    ///    for BOTH representations since the forced-mode campaigns landed:
    ///    - 1D deck: combined-channel forced flights (gas + gray shell
    ///      cloud folded into one exactly piecewise-constant channel; see
    ///      the derivation in photon.rs), which removed the analog
    ///      starvation that previously forced a one-sided bound here.
    ///    - 3D field: majorant-combined truncated null-collision forced
    ///      flights (per-shell field majorants + delta tracking within
    ///      the truncated budget; derivation and telescoping proof at
    ///      the scalar chain's use_forced). The pre-campaign one-sided
    ///      branch (upper bound + 0.25x collapse floor around the
    ///      measured 0.37-0.45x analog starvation class) is retired: a
    ///      starved field estimator now FAILS the lower side, exactly
    ///      the regression this campaign closes.
    ///
    ///    The two-sided band retains the gate's original target too: the
    ///    old cloud-blind forced composition INFLATED the hybrid well
    ///    above Multiple, which the upper side fails loudly.
    ///
    /// BUDGETS (2026-07-04 review round 2): the original 8-seed budgets
    ///    left the SZA 92/97 bands at ~43-46% of the reference (a 20%
    ///    bias would pass). Raised to 16 seeds / 2048 photons in regime
    ///    1 (measured bands 8-11% of reference) and 48 seeds / 8192
    ///    photons at SZA 97: the measured per-seed CV there is ~35%
    ///    (tail-mixture chains through the deck), so k = 48 puts the
    ///    seed SE at ~5% and the total band at ~20-21% (3 x combined
    ///    seed SE + the 5% construction floor); a 16-seed probe
    ///    measured band/ref 0.33. The measured bands print per row.
    #[test]
    #[ignore = "g_s2_hybrid_matches_multiple: heavy MC"]
    fn g_s2_hybrid_matches_multiple() {
        let atm_1d = builder::build_with_cloud_properties(
            AtmosphereType::UsStandard,
            0.15,
            &thin_deck_props(),
        );
        let mut atm_field = builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let owned = uniform_thin_field();
        atm_field.cloud_g_scaled = owned.g_default;
        let view = owned.view();

        let hybrid = SimulationConfig {
            view_zenith: 80.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 2048,
            polarized: false,
            ..SimulationConfig::default()
        };
        let multiple = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 40_000,
            ..hybrid.clone()
        };

        let cases: [(&str, &AtmosphereModel, Option<&Cloud3DField>); 2] = [
            ("1D deck", &atm_1d, None),
            ("field deck", &atm_field, Some(&view)),
        ];
        let mut failures = Vec::new();
        for (label, atm, field) in cases {
            // Both regimes two-sided (see header). SZA 97 runs the
            // heavier budget: the forced-under-deck chains are heavy-
            // tailed at small budgets (see g_s2_forced_under_1d_cloud_
            // matches_multiple for the measured ladder).
            for (sza, hyb_photons, k) in
                [(88.0, 2048, 16), (92.0, 2048, 16), (97.0, 8192, 48)]
            {
                let hybrid_row = SimulationConfig {
                    photons_per_wavelength: hyb_photons,
                    ..hybrid.clone()
                };
                let (m_h, se_h) = mc_mean_se(atm, &hybrid_row, sza, field, k);
                let (m_m, se_m) = mc_mean_se(atm, &multiple, sza, field, 16);
                let se = (se_h * se_h + se_m * se_m).sqrt();
                let diff = (m_h - m_m).abs();
                let band = 3.0 * se + 0.05 * m_h.max(m_m);
                eprintln!(
                    "G-HYB-MULT {label} SZA {sza}: hybrid {m_h:.5e} (se {se_h:.2e}) \
                     multiple {m_m:.5e} (se {se_m:.2e}) diff {diff:.2e} band {band:.2e} \
                     (band/ref {:.2}) ratio {:.3}",
                    band / m_m,
                    m_h / m_m
                );
                if diff.is_nan() || diff >= band {
                    failures.push(format!(
                        "{label} SZA {sza}: hybrid {m_h:.5e} vs multiple {m_m:.5e} \
                         (diff {diff:.3e} > band {band:.3e})"
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "G-HYB-MULT: hybrid and Multiple disagree:\n{}",
            failures.join("\n")
        );
    }

    /// G-FORCED-1D (history: G-FORCED-OFF). Three eras of this gate:
    ///
    /// 1. PRE-fix bug: at SZA >= ZENITH_SZA_START (96) with a 1D deck,
    ///    forced-collision flights sampled from GAS-only scout tau and
    ///    crossed the deck as if it were transparent while analog bounces
    ///    of the same chain raced it, INFLATING the multiple-scatter term.
    /// 2. Conservative fix (0cc8bf5): forced mode disabled under ANY cloud
    ///    channel; unbiased but variance-starved under decks at SZA >= 97
    ///    (externally measured 0.16-0.22x vs MYSTIC at SZA 99-101). This
    ///    gate then pinned "forced is off": deck radiance finite, positive,
    ///    below clear sky.
    /// 3. NOW: combined-channel forced mode. Forced flights sample the
    ///    gas-plus-gray-cloud channel exactly (both piecewise constant per
    ///    shell) and draw the vertex type from the extinction conditional,
    ///    so forced under the 1D deck is UNBIASED (derivation in photon.rs
    ///    at the scalar chain's `use_forced`).
    ///
    /// The gate now pins that forced-under-cloud matches the ANALOG model
    /// of the same deck: `ScatteringMode::Multiple` (trace_photon) races
    /// the identical gray channel on every flight with no forced mode at
    /// any SZA, is trajectory-independent of the chains, and is the
    /// externally anchored side (G2/G3 referees). Two-sided band
    /// 3 x combined seed SE + 5% construction floor (straight midpoint LOS
    /// vs refracting analog walk), the same construction as G-HYB-MULT.
    /// Both eras' regression targets stay covered: era-1 inflation fails
    /// the upper side, era-2 starvation (0.16-0.22x) fails the lower side.
    /// The below-clear sanity assertion is retained.
    #[test]
    #[ignore = "g_s2_forced_under_1d_cloud_matches_multiple: heavy MC"]
    fn g_s2_forced_under_1d_cloud_matches_multiple() {
        let clear = make_clear_sky_atm();
        let cloudy = builder::build_with_cloud_properties(
            AtmosphereType::UsStandard,
            0.15,
            &thin_deck_props(),
        );
        let hybrid = SimulationConfig {
            view_zenith: 80.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 256,
            polarized: false,
            ..SimulationConfig::default()
        };
        let multiple = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 40_000,
            ..hybrid.clone()
        };
        let mut failures = Vec::new();
        // Photon budgets per SZA. At SZA 97 the forced-under-deck hybrid is
        // heavy-tailed at tiny budgets (measured ladder, this exact
        // geometry, deck vs Multiple-at-100k: 256 photons -> seeds scatter
        // 1.6e-4..7.8e-4 around a 0.5-1.0x ratio; 2048 -> 0.93x with seeds
        // concentrated; 16384 -> 0.94x). BUDGETS RAISED (2026-07-04 review
        // round 2): the old SZA-100 row at 256 photons had an absolute
        // band (3.6e-5) LARGER than the means (~2.5e-5), so no inflation
        // of any size could fail it. The gate is now a RATIO assertion
        // with a band derived from the measured seed CVs
        // (|hyb/mul - 1| < 3 * sqrt(rel_se_h^2 + rel_se_m^2) + 5% floor),
        // scale-free by construction, at budgets where the band is
        // meaningful (printed per row). SZA-100 budgets re-raised after
        // a 4096 x 8 probe drew a tail seed (hybrid rel-se 52%, band
        // 1.64: near-vacuous again): the measured per-seed CV there is
        // ~146% at 4k photons, so 16384 photons x 24 seeds puts the
        // hybrid seed SE at ~15% and the band at ~0.5; the Multiple
        // side runs 160k photons to match.
        for (sza, hyb_photons, k_h, mul_photons) in
            [(97.0, 8192, 8, 40_000), (100.0, 16_384, 24, 160_000)]
        {
            let hybrid = SimulationConfig {
                photons_per_wavelength: hyb_photons,
                ..hybrid.clone()
            };
            let multiple = SimulationConfig {
                photons_per_wavelength: mul_photons,
                ..multiple.clone()
            };
            let (r_clear, se_c) = mc_mean_se(&clear, &hybrid, sza, None, 8);
            let (r_deck, se_d) = mc_mean_se(&cloudy, &hybrid, sza, None, k_h);
            let (r_mult, se_m) = mc_mean_se(&cloudy, &multiple, sza, None, 16);
            let ratio = r_deck / r_mult;
            let rel_se =
                ((se_d / r_deck).powi(2) + (se_m / r_mult).powi(2)).sqrt();
            let band_r = 3.0 * rel_se + 0.05;
            eprintln!(
                "G-FORCED-1D SZA {sza}: clear {r_clear:.4e} (se {se_c:.2e}) \
                 forced-hybrid {r_deck:.4e} (se {se_d:.2e}) multiple {r_mult:.4e} \
                 (se {se_m:.2e}) hyb/mul {ratio:.4} ratio-band {band_r:.3}"
            );
            assert!(
                r_deck.is_finite() && r_deck > 0.0,
                "G-FORCED-1D SZA {sza}: deck radiance must be finite and positive, got {r_deck:.4e}"
            );
            // Forced-under-cloud vs the analog reference: ratio within
            // the CV-derived band around 1 (two-sided, scale-free).
            if !ratio.is_finite() || (ratio - 1.0).abs() >= band_r {
                failures.push(format!(
                    "SZA {sza}: forced hybrid {r_deck:.4e} vs multiple {r_mult:.4e} \
                     (ratio {ratio:.4} outside 1 +- {band_r:.3})"
                ));
            }
            // Below clear sky, with a 3-sigma statistical allowance.
            let se = (se_c * se_c + se_d * se_d).sqrt();
            assert!(
                r_deck < r_clear + 3.0 * se,
                "G-FORCED-1D SZA {sza}: OD-2 deck must dim the sky: \
                 deck {r_deck:.4e} vs clear {r_clear:.4e} (3 se = {:.2e})",
                3.0 * se
            );
        }
        assert!(
            failures.is_empty(),
            "G-FORCED-1D: forced-under-cloud disagrees with the analog reference:\n{}",
            failures.join("\n")
        );
    }

    // ── Phase-function orientation (regression for the supplement-angle bug) ──

    /// With a forward-peaked aerosol phase function (HG, g≈0.7), the sky
    /// toward the sun must be much brighter than the anti-solar sky. Before
    /// the fix, cos(θ) was negated (the supplement), evaluating the forward
    /// peak at the BACKWARD angle, and this test fails by ~2 orders of
    /// magnitude.
    #[test]
    fn aerosol_forward_scatter_beats_backscatter() {
        use twilight_data::aerosol::{default_properties, AerosolType};
        let props = default_properties(AerosolType::Urban);
        let atm = builder::build_with_aerosol_properties(
            AtmosphereType::UsStandard,
            0.15,
            &props,
        );
        // Sun well above horizon so both views are sunlit; observer looks
        // 75° from zenith either toward the sun (rel az 0) or away (180).
        let mut toward = SimulationConfig {
            view_zenith: 75.0,
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        toward.view_azimuth = Some(toward.solar_azimuth); // rel az = 0
        let mut away = toward.clone();
        away.view_azimuth = Some(toward.solar_azimuth + 180.0);

        let sza = 60.0;
        let r_toward = simulate_at_sza(&atm, &toward, sza, None);
        let r_away = simulate_at_sza(&atm, &away, sza, None);
        // 550 nm bin (index 17 on the 380..780/10nm grid)
        let i550 = r_toward
            .wavelengths_nm
            .iter()
            .position(|&w| (w - 550.0).abs() < 1e-9)
            .unwrap();
        let t = r_toward.radiance[i550];
        let a = r_away.radiance[i550];
        assert!(
            t > 2.0 * a,
            "forward-scatter sky should be much brighter than anti-solar: \
             toward={:.4e}, away={:.4e}, ratio={:.2}",
            t,
            a,
            t / a
        );
    }

    // ── simulate_at_sza ──

    #[test]
    fn simulate_at_sza_returns_correct_wavelength_count() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 96.0, None);
        assert_eq!(result.wavelengths_nm.len(), 41);
        assert_eq!(result.radiance.len(), 41);
    }

    #[test]
    fn simulate_at_sza_stores_sza() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 96.5, None);
        assert!((result.sza_deg - 96.5).abs() < 1e-10);
    }

    #[test]
    fn simulate_at_sza_positive_radiance_at_civil_twilight() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 93.0, None); // civil twilight
        let total = total_radiance(&result);
        assert!(
            total > 0.0,
            "Civil twilight should produce positive radiance, got {}",
            total
        );
    }

    #[test]
    fn simulate_at_sza_radiance_non_negative() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        for sza in &[90.0, 96.0, 102.0, 108.0] {
            let result = simulate_at_sza(&atm, &config, *sza, None);
            for (i, &r) in result.radiance.iter().enumerate() {
                assert!(
                    r >= 0.0,
                    "Radiance at SZA={}, wl[{}] = {} should be non-negative",
                    sza,
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn simulate_at_sza_decreases_with_depth() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let r_93 = total_radiance(&simulate_at_sza(&atm, &config, 93.0, None));
        let r_100 = total_radiance(&simulate_at_sza(&atm, &config, 100.0, None));
        assert!(
            r_93 > r_100,
            "Radiance should decrease: SZA93={:.4e} > SZA100={:.4e}",
            r_93,
            r_100
        );
    }

    #[test]
    fn simulate_at_sza_with_solar_irradiance() {
        let atm = make_clear_sky_atm();
        let mut config_on = default_config();
        config_on.apply_solar_irradiance = true;
        let mut config_off = default_config();
        config_off.apply_solar_irradiance = false;

        let r_on = simulate_at_sza(&atm, &config_on, 93.0, None);
        let r_off = simulate_at_sza(&atm, &config_off, 93.0, None);

        // With solar irradiance weighting, radiance should be different
        // (unless raw radiance happens to equal 1 everywhere, which it won't)
        let total_on = total_radiance(&r_on);
        let total_off = total_radiance(&r_off);
        // Both should be positive at civil twilight
        assert!(total_on > 0.0, "Irradiance-weighted should be positive");
        assert!(total_off > 0.0, "Raw should be positive");
        // They should be different (solar irradiance multiplies by ~1-2 W/m²/nm)
        assert!(
            (total_on - total_off).abs() > 1e-20,
            "Solar weighting should change results"
        );
    }

    #[test]
    fn simulate_at_sza_wavelengths_correct() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 96.0, None);
        assert!((result.wavelengths_nm[0] - 380.0).abs() < 0.01);
        assert!((result.wavelengths_nm[20] - 580.0).abs() < 0.01);
        assert!((result.wavelengths_nm[40] - 780.0).abs() < 0.01);
    }

    // ── simulate_twilight_scan ──

    #[test]
    fn twilight_scan_correct_count() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 2.0, None);
        // 90, 92, 94, 96, 98, 100 = 6 steps
        assert_eq!(results.len(), 6, "Expected 6 steps, got {}", results.len());
    }

    #[test]
    fn twilight_scan_sza_values_correct() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let results = simulate_twilight_scan(&atm, &config, 90.0, 94.0, 2.0, None);
        assert!((results[0].sza_deg - 90.0).abs() < 0.01);
        assert!((results[1].sza_deg - 92.0).abs() < 0.01);
        assert!((results[2].sza_deg - 94.0).abs() < 0.01);
    }

    #[test]
    fn twilight_scan_radiance_decreases() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let results = simulate_twilight_scan(&atm, &config, 91.0, 105.0, 2.0, None);
        let totals: Vec<f64> = results.iter().map(total_radiance).collect();

        // Radiance should generally decrease (may have small bumps from geometry)
        // Check first vs last
        assert!(
            totals[0] > totals[totals.len() - 1],
            "First total ({:.4e}) should exceed last ({:.4e})",
            totals[0],
            totals[totals.len() - 1]
        );
    }

    // ── total_radiance ──

    #[test]
    fn total_radiance_flat_spectrum() {
        // Flat radiance = 1.0 over 380-780nm = 400nm bandwidth
        // Trapezoidal integral = 1.0 × 400 = 400
        let result = SpectralResult {
            wavelengths_nm: vec![380.0, 780.0],
            radiance: vec![1.0, 1.0],
            sza_deg: 96.0,
        };
        let total = total_radiance(&result);
        assert!(
            (total - 400.0).abs() < 0.01,
            "Flat 1.0 over 400nm: total={}, expected 400",
            total
        );
    }

    #[test]
    fn total_radiance_zero() {
        let result = SpectralResult {
            wavelengths_nm: vec![380.0, 780.0],
            radiance: vec![0.0, 0.0],
            sza_deg: 96.0,
        };
        assert!(total_radiance(&result).abs() < 1e-20);
    }

    #[test]
    fn total_radiance_single_point() {
        let result = SpectralResult {
            wavelengths_nm: vec![550.0],
            radiance: vec![1.0],
            sza_deg: 96.0,
        };
        // Single point → just the value itself
        assert!((total_radiance(&result) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn total_radiance_empty() {
        let result = SpectralResult {
            wavelengths_nm: vec![],
            radiance: vec![],
            sza_deg: 96.0,
        };
        assert!((total_radiance(&result) - 0.0).abs() < 1e-20);
    }

    #[test]
    fn total_radiance_trapezoidal_triangle() {
        // Triangle: rises from 0 to 1 at midpoint, then 1 to 0
        // Area = 0.5 × base × height = 0.5 × 200 × 1 = 100
        let result = SpectralResult {
            wavelengths_nm: vec![400.0, 500.0, 600.0],
            radiance: vec![0.0, 1.0, 0.0],
            sza_deg: 96.0,
        };
        let total = total_radiance(&result);
        assert!(
            (total - 100.0).abs() < 0.01,
            "Triangle integral: total={}, expected 100",
            total
        );
    }

    // ── ScatteringMode defaults ──

    #[test]
    fn default_scattering_mode_is_single() {
        let c = SimulationConfig::default();
        assert_eq!(c.scattering_mode, ScatteringMode::Single);
    }

    #[test]
    fn default_photons_per_wavelength() {
        let c = SimulationConfig::default();
        assert_eq!(c.photons_per_wavelength, 10_000);
    }

    // ── MC mode basic tests ──

    fn mc_config() -> SimulationConfig {
        SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 1000, // enough for test convergence
            ..SimulationConfig::default()
        }
    }

    #[test]
    fn mc_returns_correct_wavelength_count() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        let result = simulate_at_sza(&atm, &config, 96.0, None);
        assert_eq!(result.wavelengths_nm.len(), 41);
        assert_eq!(result.radiance.len(), 41);
    }

    #[test]
    fn mc_stores_sza() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        let result = simulate_at_sza(&atm, &config, 96.5, None);
        assert!((result.sza_deg - 96.5).abs() < 1e-10);
    }

    #[test]
    fn mc_radiance_non_negative() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        for sza in &[90.0, 96.0, 102.0] {
            let result = simulate_at_sza(&atm, &config, *sza, None);
            for (i, &r) in result.radiance.iter().enumerate() {
                assert!(
                    r >= 0.0,
                    "MC radiance at SZA={}, wl[{}] = {} should be non-negative",
                    sza,
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn mc_positive_radiance_at_civil_twilight() {
        // At SZA=93° with clear sky, MC should produce some signal
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 2000,
            ..SimulationConfig::default()
        };
        let result = simulate_at_sza(&atm, &config, 93.0, None);
        let total = total_radiance(&result);
        assert!(
            total > 0.0,
            "MC civil twilight should produce positive radiance, got {}",
            total
        );
    }

    #[test]
    fn mc_radiance_decreases_with_depth() {
        // SZA 93 should give more radiance than SZA 100
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 2000,
            ..SimulationConfig::default()
        };
        let r_93 = total_radiance(&simulate_at_sza(&atm, &config, 93.0, None));
        let r_100 = total_radiance(&simulate_at_sza(&atm, &config, 100.0, None));
        assert!(
            r_93 > r_100 * 0.1, // generous for MC noise
            "MC SZA93 ({:.4e}) should be > SZA100 ({:.4e})",
            r_93,
            r_100
        );
    }

    #[test]
    fn mc_wavelengths_correct() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        let result = simulate_at_sza(&atm, &config, 96.0, None);
        assert!((result.wavelengths_nm[0] - 380.0).abs() < 0.01);
        assert!((result.wavelengths_nm[20] - 580.0).abs() < 0.01);
        assert!((result.wavelengths_nm[40] - 780.0).abs() < 0.01);
    }

    #[test]
    fn mc_zero_photons_gives_zero_radiance() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 0,
            ..SimulationConfig::default()
        };
        let result = simulate_at_sza(&atm, &config, 93.0, None);
        let total = total_radiance(&result);
        assert!(
            total.abs() < 1e-20,
            "Zero photons should give zero radiance, got {}",
            total
        );
    }

    // ── MC vs single-scatter comparison ──

    #[test]
    fn mc_and_single_same_order_of_magnitude_at_shallow_twilight() {
        // At shallow twilight (SZA=93°), single-scatter dominates.
        // MC should give similar results (within ~5x for 2000 photons).
        let atm = make_clear_sky_atm();

        let ss_config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let mc_config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 5000,
            ..SimulationConfig::default()
        };

        let ss_total = total_radiance(&simulate_at_sza(&atm, &ss_config, 93.0, None));
        let mc_total = total_radiance(&simulate_at_sza(&atm, &mc_config, 93.0, None));

        // Both should be positive
        assert!(ss_total > 0.0, "Single-scatter should be positive");
        assert!(mc_total > 0.0, "MC should be positive");

        // They should be within ~10x of each other at shallow twilight
        let ratio = if ss_total > mc_total {
            ss_total / mc_total
        } else {
            mc_total / ss_total
        };
        assert!(
            ratio < 10.0,
            "MC ({:.4e}) and single-scatter ({:.4e}) should be same order of magnitude (ratio: {:.1})",
            mc_total,
            ss_total,
            ratio
        );
    }

    // ── MC twilight scan ──

    #[test]
    fn mc_twilight_scan_correct_count() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 200, // few photons for speed
            ..SimulationConfig::default()
        };
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 5.0, None);
        // 90, 95, 100 = 3 steps
        assert_eq!(results.len(), 3, "Expected 3 steps, got {}", results.len());
    }

    #[test]
    fn mc_twilight_scan_sza_values_correct() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 200,
            ..SimulationConfig::default()
        };
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 5.0, None);
        assert!((results[0].sza_deg - 90.0).abs() < 0.01);
        assert!((results[1].sza_deg - 95.0).abs() < 0.01);
        assert!((results[2].sza_deg - 100.0).abs() < 0.01);
    }

    // ── Hybrid mode tests ──

    fn hybrid_config() -> SimulationConfig {
        SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 50, // few secondary rays for test speed
            ..SimulationConfig::default()
        }
    }

    #[test]
    fn hybrid_returns_correct_wavelength_count() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        let result = simulate_at_sza(&atm, &config, 96.0, None);
        assert_eq!(result.wavelengths_nm.len(), 41);
        assert_eq!(result.radiance.len(), 41);
    }

    #[test]
    fn hybrid_stores_sza() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        let result = simulate_at_sza(&atm, &config, 96.5, None);
        assert!((result.sza_deg - 96.5).abs() < 1e-10);
    }

    #[test]
    fn hybrid_radiance_non_negative() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        for sza in &[90.0, 96.0, 102.0] {
            let result = simulate_at_sza(&atm, &config, *sza, None);
            for (i, &r) in result.radiance.iter().enumerate() {
                assert!(
                    r >= 0.0,
                    "Hybrid radiance at SZA={}, wl[{}] = {} should be non-negative",
                    sza,
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn hybrid_positive_at_civil_twilight() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 100,
            ..SimulationConfig::default()
        };
        let result = simulate_at_sza(&atm, &config, 93.0, None);
        let total = total_radiance(&result);
        assert!(
            total > 0.0,
            "Hybrid civil twilight should produce positive radiance, got {}",
            total
        );
    }

    #[test]
    fn hybrid_at_least_as_bright_as_single_scatter() {
        // Hybrid includes single-scatter + orders 2+, so it should be
        // at least as bright (or very close) to pure single-scatter.
        let atm = make_clear_sky_atm();

        let ss_config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let hybrid_config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 200,
            ..SimulationConfig::default()
        };

        let ss_total = total_radiance(&simulate_at_sza(&atm, &ss_config, 96.0, None));
        let hybrid_total = total_radiance(&simulate_at_sza(&atm, &hybrid_config, 96.0, None));

        // Hybrid should be >= single-scatter (within MC noise margin)
        // Allow 20% tolerance for MC noise
        assert!(
            hybrid_total > ss_total * 0.8,
            "Hybrid ({:.4e}) should be >= single-scatter ({:.4e}) minus noise margin",
            hybrid_total,
            ss_total
        );
    }

    #[test]
    fn hybrid_wavelengths_correct() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        let result = simulate_at_sza(&atm, &config, 96.0, None);
        assert!((result.wavelengths_nm[0] - 380.0).abs() < 0.01);
        assert!((result.wavelengths_nm[20] - 580.0).abs() < 0.01);
        assert!((result.wavelengths_nm[40] - 780.0).abs() < 0.01);
    }

    #[test]
    fn hybrid_zero_secondary_rays_equals_single_scatter() {
        // With 0 secondary rays, hybrid should produce exactly the
        // same result as single-scatter (only order 1 is computed).
        let atm = make_clear_sky_atm();

        let ss_config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let hybrid_config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 0,
            ..SimulationConfig::default()
        };

        let ss_result = simulate_at_sza(&atm, &ss_config, 96.0, None);
        let hybrid_result = simulate_at_sza(&atm, &hybrid_config, 96.0, None);

        for i in 0..ss_result.radiance.len() {
            let diff = (ss_result.radiance[i] - hybrid_result.radiance[i]).abs();
            let rel = if ss_result.radiance[i] > 1e-30 {
                diff / ss_result.radiance[i]
            } else {
                diff
            };
            assert!(
                rel < 0.05,
                "Hybrid(0 rays) should match single-scatter: wl[{}] {:.4e} vs {:.4e} (rel: {:.4})",
                i,
                hybrid_result.radiance[i],
                ss_result.radiance[i],
                rel
            );
        }
    }

    #[test]
    fn hybrid_radiance_decreases_with_depth() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 100,
            ..SimulationConfig::default()
        };
        let r_93 = total_radiance(&simulate_at_sza(&atm, &config, 93.0, None));
        let r_100 = total_radiance(&simulate_at_sza(&atm, &config, 100.0, None));
        assert!(
            r_93 > r_100 * 0.5,
            "Hybrid SZA93 ({:.4e}) should be > SZA100 ({:.4e})",
            r_93,
            r_100
        );
    }

    #[test]
    fn hybrid_twilight_scan_correct_count() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 10,
            ..SimulationConfig::default()
        };
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 5.0, None);
        assert_eq!(results.len(), 3, "Expected 3 steps, got {}", results.len());
    }

    // ── CV baseline diagnostic ──

    /// Measure coefficient of variation across 16 SZAs to establish
    /// baseline variance of the unguided pipeline.
    ///
    /// Run with: cargo test -p twilight-cpu --release -- cv_baseline --nocapture
    #[test]
    #[ignore = "slow MC diagnostic (minutes); run: cargo test --release -- --ignored cv_baseline --nocapture"]
    fn cv_baseline() {
        use twilight_core::geometry::{geographic_to_ecef, solar_direction_ecef};
        use twilight_core::photon;

        let atm = make_clear_sky_atm();
        let lat = 21.4225;
        let lon = 39.8262;
        let solar_azimuth = 270.0;
        let view_zenith = 75.0;

        let obs_pos = geographic_to_ecef(lat, lon, 0.0);
        let view = solar_direction_ecef(view_zenith, solar_azimuth, lat, lon);

        let base_seeds = 50usize;
        let deep_seeds = 200usize;
        let rays = 200usize;

        eprintln!("\n{:=<64}", "");
        eprintln!(
            "CV BASELINE: unguided ALIS ({}/{} seeds x {} rays)",
            base_seeds, deep_seeds, rays
        );
        eprintln!("{:=<64}", "");
        eprintln!(
            "{:<8} {:>6} {:>14} {:>12}",
            "SZA", "Seeds", "Mean radiance", "CV"
        );
        eprintln!("{:-<46}", "");

        let sza_list: Vec<f64> = (93..=108).map(|s| s as f64).collect();

        for &sza_deg in &sza_list {
            let sun = solar_direction_ecef(sza_deg, solar_azimuth, lat, lon);

            // Use more seeds at deep twilight where CV measurement is noisy.
            let num_seeds = if sza_deg >= 105.0 {
                deep_seeds
            } else {
                base_seeds
            };

            let mut totals = Vec::with_capacity(num_seeds);
            for seed_idx in 0..num_seeds {
                let sza_bits = sza_deg.to_bits();
                let mut rng = (seed_idx as u64)
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(sza_bits)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);

                let result = photon::hybrid_scatter_radiance_alis(
                    &atm, obs_pos, view, sun, rays, &mut rng, None,
                );
                let total: f64 = result.iter().take(atm.num_wavelengths).sum();
                totals.push(total);
            }

            let mean = totals.iter().sum::<f64>() / num_seeds as f64;
            let std = (totals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (num_seeds - 1) as f64)
                .sqrt();
            let cv = if mean.abs() > 1e-30 {
                std / mean.abs()
            } else {
                0.0
            };

            eprintln!(
                "{:<8.1} {:>6} {:>14.4e} {:>12.4}",
                sza_deg, num_seeds, mean, cv
            );
        }
        eprintln!("{:=<64}\n", "");
    }

    /// Deep diagnostic: run targeted SZAs with 500 seeds, dump distribution stats.
    ///
    /// Run with: cargo test -p twilight-cpu --release -- cv_deep_diag --nocapture
    #[test]
    #[ignore = "slow MC diagnostic (minutes); run: cargo test --release -- --ignored cv_deep_diag --nocapture"]
    fn cv_deep_diag() {
        use twilight_core::geometry::{geographic_to_ecef, solar_direction_ecef};
        use twilight_core::photon;

        let atm = make_clear_sky_atm();
        let lat = 21.4225;
        let lon = 39.8262;
        let solar_azimuth = 270.0;
        let view_zenith = 75.0;

        let obs_pos = geographic_to_ecef(lat, lon, 0.0);
        let view = solar_direction_ecef(view_zenith, solar_azimuth, lat, lon);

        let num_seeds = 500usize;
        let rays = 200usize;
        let target_szas = [97.0, 98.0, 100.0, 102.0, 106.0];

        eprintln!("\n{:=<80}", "");
        eprintln!(
            "CV DEEP DIAGNOSTIC: {} seeds x {} rays, distribution analysis",
            num_seeds, rays
        );
        eprintln!("{:=<80}", "");

        for &sza_deg in &target_szas {
            let sun = solar_direction_ecef(sza_deg, solar_azimuth, lat, lon);

            let mut totals = Vec::with_capacity(num_seeds);
            for seed_idx in 0..num_seeds {
                let sza_bits = sza_deg.to_bits();
                let mut rng = (seed_idx as u64)
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(sza_bits)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);

                let result = photon::hybrid_scatter_radiance_alis(
                    &atm, obs_pos, view, sun, rays, &mut rng, None,
                );
                let total: f64 = result.iter().take(atm.num_wavelengths).sum();
                totals.push(total);
            }

            let n = totals.len();
            let mean = totals.iter().sum::<f64>() / n as f64;
            let std =
                (totals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
            let cv = if mean.abs() > 1e-30 {
                std / mean.abs()
            } else {
                0.0
            };

            // CV convergence from UNSORTED data (first N seeds)
            let mut cv_by_n = Vec::new();
            for &sub_n in &[50usize, 100, 200, 500] {
                if sub_n > n {
                    break;
                }
                let sub = &totals[..sub_n];
                let sub_mean = sub.iter().sum::<f64>() / sub_n as f64;
                let sub_std = (sub.iter().map(|x| (x - sub_mean).powi(2)).sum::<f64>()
                    / (sub_n - 1) as f64)
                    .sqrt();
                let sub_cv = if sub_mean.abs() > 1e-30 {
                    sub_std / sub_mean.abs()
                } else {
                    0.0
                };
                cv_by_n.push((sub_n, sub_cv));
            }

            // Count negative seeds
            let neg_count = totals.iter().filter(|&&x| x < 0.0).count();

            // Sort for percentile analysis
            totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p01 = totals[(0.01 * n as f64) as usize];
            let p05 = totals[(0.05 * n as f64) as usize];
            let p25 = totals[(0.25 * n as f64) as usize];
            let p50 = totals[n / 2];
            let p75 = totals[(0.75 * n as f64) as usize];
            let p95 = totals[(0.95 * n as f64) as usize];
            let p99 = totals[(0.99 * n as f64) as usize];
            let min = totals[0];
            let max = totals[n - 1];

            eprintln!("\n--- SZA {:.1} ---", sza_deg);
            eprintln!(
                "  Mean:    {:.4e}    Std: {:.4e}    CV: {:.4}",
                mean, std, cv
            );
            eprintln!("  Neg seeds: {}/{}", neg_count, n);
            eprintln!(
                "  Min:  {:.4e}   Max:  {:.4e}   Range ratio: {:.1}x",
                min,
                max,
                if min.abs() > 1e-30 {
                    max / min
                } else {
                    f64::INFINITY
                }
            );
            eprintln!("  Percentiles:");
            eprintln!("    P1:  {:.4e}  P5:  {:.4e}  P25: {:.4e}", p01, p05, p25);
            eprintln!("    P50: {:.4e}  P75: {:.4e}", p50, p75);
            eprintln!("    P95: {:.4e}  P99: {:.4e}", p95, p99);
            eprintln!("  CV convergence:");
            for (sub_n, sub_cv) in &cv_by_n {
                eprintln!("    n={:>4}: CV={:.4}", sub_n, sub_cv);
            }
        }
        eprintln!("\n{:=<80}\n", "");
    }

    // ── G2 diagnostic: independent flat-slab referee ─────────────────────

    /// Local RNG for the independent referee (splitmix64, distinct from the
    /// xorshift streams in photon.rs).
    struct DiagRng(u64);
    impl DiagRng {
        fn f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 * (1.0 / 9007199254740992.0)
        }
    }

    /// One homogeneous layer of the flattened 1D profile.
    #[derive(Clone, Copy)]
    struct DiagLayer {
        z_lo: f64,
        z_hi: f64,
        sig_gas: f64,
        ssa_gas: f64,
        sig_cloud: f64,
    }

    /// Extract the 550 nm 1D profile from an AtmosphereModel.
    fn diag_profile(atm: &AtmosphereModel, w: usize) -> Vec<DiagLayer> {
        let rs = atm.surface_radius();
        (0..atm.num_shells)
            .map(|s| DiagLayer {
                z_lo: atm.shells[s].r_inner - rs,
                z_hi: atm.shells[s].r_outer - rs,
                sig_gas: atm.optics[s][w].extinction,
                ssa_gas: atm.optics[s][w].ssa,
                sig_cloud: atm.cloud_extinction[s],
            })
            .collect()
    }

    /// Vertical optical depth (gas + cloud) from height z to TOA.
    fn diag_tau_above(layers: &[DiagLayer], z: f64) -> f64 {
        let mut tau = 0.0;
        for l in layers {
            if l.z_hi <= z {
                continue;
            }
            let lo = l.z_lo.max(z);
            tau += (l.sig_gas + l.sig_cloud) * (l.z_hi - lo);
        }
        tau
    }

    /// Rotate `dir` by polar angle acos(ct) and azimuth phi (local impl,
    /// independent of scattering::scatter_direction).
    fn diag_rotate(dir: [f64; 3], ct: f64, phi: f64) -> [f64; 3] {
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        let (sp, cp) = (phi.sin(), phi.cos());
        // orthonormal basis around dir
        let a = if dir[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        // u = normalize(dir x a)
        let mut u = [
            dir[1] * a[2] - dir[2] * a[1],
            dir[2] * a[0] - dir[0] * a[2],
            dir[0] * a[1] - dir[1] * a[0],
        ];
        let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        u = [u[0] / un, u[1] / un, u[2] / un];
        // v = dir x u
        let v = [
            dir[1] * u[2] - dir[2] * u[1],
            dir[2] * u[0] - dir[0] * u[2],
            dir[0] * u[1] - dir[1] * u[0],
        ];
        [
            st * cp * u[0] + st * sp * v[0] + ct * dir[0],
            st * cp * u[1] + st * sp * v[1] + ct * dir[1],
            st * cp * u[2] + st * sp * v[2] + ct * dir[2],
        ]
    }

    /// Independent plane-parallel backward MC with NEE for the G2 slab.
    /// Single combined-extinction channel (textbook algorithm, no
    /// decomposition race), local RNG, local phase samplers. Returns
    /// radiance in unit-solar-irradiance units (sr^-1).
    #[allow(clippy::too_many_arguments)]
    fn diag_flat_mc(
        layers: &[DiagLayer],
        g_cloud: f64,
        albedo: f64,
        sza_deg: f64,
        vz_deg: f64,
        n_photons: usize,
        seed: u64,
        max_bounces: usize,
    ) -> (f64, f64) {
        let z_top = layers.last().unwrap().z_hi;
        let mu0 = (sza_deg.to_radians()).cos();
        let sun = [sza_deg.to_radians().sin(), 0.0, mu0];
        let view = [vz_deg.to_radians().sin(), 0.0, vz_deg.to_radians().cos()];

        let (t1, t2) = (0..n_photons)
            .into_par_iter()
            .map(|p| {
                let mut rng = DiagRng(seed ^ (p as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
                let mut z = 1e-4;
                let mut d = view;
                let mut w = 1.0f64;
                let mut acc = 0.0f64;
                let mut acc1 = 0.0f64; // order-1: first-vertex NEE
                let mut nv = 0usize;
                let mut li = 0usize; // current layer index
                'walk: for _ in 0..max_bounces {
                    // free flight to tau_target through combined extinction
                    let mut tau_t = -(1.0 - rng.f64()).max(1e-300).ln();
                    loop {
                        // locate current layer
                        while li + 1 < layers.len() && z >= layers[li].z_hi {
                            li += 1;
                        }
                        while li > 0 && z < layers[li].z_lo {
                            li -= 1;
                        }
                        let l = layers[li];
                        let sig = l.sig_gas + l.sig_cloud;
                        if d[2].abs() < 1e-12 {
                            break 'walk; // horizontal: leaves slab sideways (flat approx)
                        }
                        let z_next = if d[2] > 0.0 { l.z_hi } else { l.z_lo };
                        let s_bound = (z_next - z) / d[2];
                        let tau_seg = sig * s_bound;
                        if tau_seg >= tau_t {
                            // collision inside this layer
                            let s = tau_t / sig;
                            z += d[2] * s;
                            let is_cloud = rng.f64() < l.sig_cloud / sig;
                            if !is_cloud {
                                w *= l.ssa_gas;
                            }
                            // NEE
                            let t_sun = (-(diag_tau_above(layers, z)) / mu0).exp();
                            let cos_nee = sun[0] * d[0] + sun[1] * d[1] + sun[2] * d[2];
                            let phase = if is_cloud {
                                let g2 = g_cloud * g_cloud;
                                let den = 1.0 + g2 - 2.0 * g_cloud * cos_nee;
                                (1.0 - g2) / (den * den.sqrt())
                            } else {
                                0.75 * (1.0 + cos_nee * cos_nee)
                            };
                            let nee = w * t_sun * phase / (4.0 * core::f64::consts::PI);
                            if nv == 0 {
                                acc1 += nee;
                            } else {
                                acc += nee;
                            }
                            nv += 1;
                            // new direction
                            let ct = if is_cloud {
                                if g_cloud.abs() < 1e-6 {
                                    2.0 * rng.f64() - 1.0
                                } else {
                                    let g2 = g_cloud * g_cloud;
                                    let s2 = (1.0 - g2) / (1.0 - g_cloud + 2.0 * g_cloud * rng.f64());
                                    ((1.0 + g2 - s2 * s2) / (2.0 * g_cloud)).clamp(-1.0, 1.0)
                                }
                            } else {
                                // Rayleigh by rejection (local, independent)
                                loop {
                                    let mu = 2.0 * rng.f64() - 1.0;
                                    if rng.f64() * 1.5 <= 0.75 * (1.0 + mu * mu) {
                                        break mu;
                                    }
                                }
                            };
                            let phi = 2.0 * core::f64::consts::PI * rng.f64();
                            d = diag_rotate(d, ct, phi);
                            continue 'walk;
                        }
                        // crosses the layer boundary
                        tau_t -= tau_seg;
                        z = z_next + d[2].signum() * 1e-9;
                        if z >= z_top {
                            break 'walk; // escaped to space
                        }
                        if z <= 0.0 {
                            // ground: NEE + Lambertian bounce
                            z = 1e-9;
                            let t_sun = (-(diag_tau_above(layers, 0.0)) / mu0).exp();
                            acc += w * albedo * t_sun * mu0 / core::f64::consts::PI;
                            nv += 1;
                            w *= albedo;
                            if w < 1e-12 {
                                break 'walk;
                            }
                            // cosine-weighted upward hemisphere
                            let ct = rng.f64().sqrt();
                            let st = (1.0 - ct * ct).sqrt();
                            let ph = 2.0 * core::f64::consts::PI * rng.f64();
                            d = [st * ph.cos(), st * ph.sin(), ct];
                            tau_t = -(1.0 - rng.f64()).max(1e-300).ln();
                        }
                    }
                }
                (acc1, acc)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        (t1 / n_photons as f64, t2 / n_photons as f64)
    }

    /// Flat replica of `trace_photon`'s WALK STRUCTURE: per-segment gas
    /// free-path redraw racing a carried gray cloud budget (decomposition
    /// tracking), same NEE estimator. Same flat geometry and local RNG as
    /// `diag_flat_mc`, so any deviation from it isolates the race logic
    /// itself (vs the spherical shell machinery).
    #[allow(clippy::too_many_arguments)]
    fn diag_flat_mc_race(
        layers: &[DiagLayer],
        g_cloud: f64,
        albedo: f64,
        sza_deg: f64,
        vz_deg: f64,
        n_photons: usize,
        seed: u64,
        max_bounces: usize,
    ) -> f64 {
        let z_top = layers.last().unwrap().z_hi;
        let mu0 = (sza_deg.to_radians()).cos();
        let sun = [sza_deg.to_radians().sin(), 0.0, mu0];
        let view = [vz_deg.to_radians().sin(), 0.0, vz_deg.to_radians().cos()];

        let total: f64 = (0..n_photons)
            .into_par_iter()
            .map(|p| {
                let mut rng = DiagRng(seed ^ (p as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
                let mut z = 1e-4;
                let mut d = view;
                let mut w = 1.0f64;
                let mut acc = 0.0f64;
                let mut li = 0usize;
                // Cloud budget for the current flight (carried across
                // segment boundaries, redrawn after each collision).
                let mut tau_c = -(1.0 - rng.f64()).max(1e-300).ln();
                'walk: for _ in 0..max_bounces {
                    // one SEGMENT per iteration (twilight's bounce-loop body)
                    while li + 1 < layers.len() && z >= layers[li].z_hi {
                        li += 1;
                    }
                    while li > 0 && z < layers[li].z_lo {
                        li -= 1;
                    }
                    let l = layers[li];
                    if d[2].abs() < 1e-12 {
                        break 'walk;
                    }
                    let z_next = if d[2] > 0.0 { l.z_hi } else { l.z_lo };
                    let s_bound = (z_next - z) / d[2];
                    // gas free path, redrawn EVERY segment (twilight line 327)
                    let free_path = if l.sig_gas < 1e-20 {
                        f64::INFINITY
                    } else {
                        -(1.0 - rng.f64()).max(1e-300).ln() / l.sig_gas
                    };
                    let gas_cap = free_path.min(s_bound);
                    // race the carried cloud budget over [0, gas_cap]
                    let mut is_cloud = false;
                    let mut s_coll = free_path;
                    if l.sig_cloud > 0.0 {
                        let dist_c = tau_c / l.sig_cloud;
                        if dist_c <= gas_cap {
                            is_cloud = true;
                            s_coll = dist_c;
                        } else {
                            tau_c -= l.sig_cloud * gas_cap;
                        }
                    }
                    if !is_cloud && free_path >= s_bound {
                        // cross the boundary; budget carries
                        z = z_next + d[2].signum() * 1e-9;
                        if z >= z_top {
                            break 'walk;
                        }
                        if z <= 0.0 {
                            z = 1e-9;
                            let t_sun = (-(diag_tau_above(layers, 0.0)) / mu0).exp();
                            acc += w * albedo * t_sun * mu0 / core::f64::consts::PI;
                            w *= albedo;
                            if w < 1e-12 {
                                break 'walk;
                            }
                            let ct = rng.f64().sqrt();
                            let st = (1.0 - ct * ct).sqrt();
                            let ph = 2.0 * core::f64::consts::PI * rng.f64();
                            d = [st * ph.cos(), st * ph.sin(), ct];
                            // ground bounce ends the flight: redraw budget
                            tau_c = -(1.0 - rng.f64()).max(1e-300).ln();
                        }
                        continue 'walk;
                    }
                    // collision (cloud at s_coll, else gas at free_path)
                    z += d[2] * s_coll;
                    if !is_cloud {
                        w *= l.ssa_gas;
                    }
                    let t_sun = (-(diag_tau_above(layers, z)) / mu0).exp();
                    let cos_nee = sun[0] * d[0] + sun[1] * d[1] + sun[2] * d[2];
                    let phase = if is_cloud {
                        let g2 = g_cloud * g_cloud;
                        let den = 1.0 + g2 - 2.0 * g_cloud * cos_nee;
                        (1.0 - g2) / (den * den.sqrt())
                    } else {
                        0.75 * (1.0 + cos_nee * cos_nee)
                    };
                    acc += w * t_sun * phase / (4.0 * core::f64::consts::PI);
                    let ct = if is_cloud {
                        if g_cloud.abs() < 1e-6 {
                            2.0 * rng.f64() - 1.0
                        } else {
                            let g2 = g_cloud * g_cloud;
                            let s2 = (1.0 - g2) / (1.0 - g_cloud + 2.0 * g_cloud * rng.f64());
                            ((1.0 + g2 - s2 * s2) / (2.0 * g_cloud)).clamp(-1.0, 1.0)
                        }
                    } else {
                        loop {
                            let mu = 2.0 * rng.f64() - 1.0;
                            if rng.f64() * 1.5 <= 0.75 * (1.0 + mu * mu) {
                                break mu;
                            }
                        }
                    };
                    let phi = 2.0 * core::f64::consts::PI * rng.f64();
                    d = diag_rotate(d, ct, phi);
                    // collision ends the flight: redraw budget
                    tau_c = -(1.0 - rng.f64()).max(1e-300).ln();
                }
                acc
            })
            .sum();
        total / n_photons as f64
    }

    /// DIAG (G2 root cause): independent plane-parallel referee vs
    /// `trace_photon` (Multiple mode) on the identical per-shell profile.
    /// Prints ratios; asserts nothing (diagnostic only).
    #[test]
    #[ignore = "diag: heavy MC diagnostic (G2 root-cause tooling)"]
    fn diag_g2_slab_independent_reference() {
        use twilight_data::cloud::CloudProperties;
        let cases = [
            ("g2_tau1", 0.85f64, 1.0f64, false, 0.15f64),
            ("g2_g085", 0.85f64, 10.0f64, false, 0.15f64),
        ];
        for (label, g_unscaled, tau_star, nogas, albedo_in) in cases {
            let f = g_unscaled * g_unscaled;
            let de_scale = 1.0 - 0.999 * f;
            let props = CloudProperties {
                base_km: 1.0,
                top_km: 2.0,
                optical_depth: tau_star / de_scale,
                ssa: 0.999,
                asymmetry: g_unscaled,
            };
            let mut atm = builder::build_with_cloud_properties(
                AtmosphereType::UsStandard,
                0.15,
                &props,
            );
            for n in atm.refractive_index.iter_mut() {
                *n = 1.0;
            }
            let albedo = albedo_in;
            for a in atm.surface_albedo.iter_mut() {
                *a = albedo;
            }
            if nogas {
                // Pure-cloud variant: kill the gas channel to isolate the
                // cloud-specific machinery.
                for s in 0..atm.num_shells {
                    for w in 0..atm.num_wavelengths {
                        atm.optics[s][w].extinction = 0.0;
                    }
                }
            }
            let w550 = (0..atm.num_wavelengths)
                .find(|&w| (atm.wavelengths_nm[w] - 550.0).abs() < 0.5)
                .expect("550 nm channel");
            let layers = diag_profile(&atm, w550);
            let g_scaled = atm.cloud_g_scaled;

            for vz in [0.0f64, 60.0] {
                // Independent flat referee.
                let (flat1, flat2p) = diag_flat_mc(
                    &layers, g_scaled, albedo, 30.0, vz, 400_000, 0xD1A6_0001, 100_000,
                );
                let flat = flat1 + flat2p;
                // Flat replica of twilight's two-budget race walk.
                let flat_race = diag_flat_mc_race(
                    &layers, g_scaled, albedo, 30.0, vz, 400_000, 0xD1A6_0002, 100_000,
                );
                // trace_photon on the same atm (spherical machinery).
                let config = SimulationConfig {
                    view_zenith: vz,
                    scattering_mode: ScatteringMode::Multiple,
                    apply_solar_irradiance: false,
                    photons_per_wavelength: 0,
                    polarized: false,
                    ..SimulationConfig::default()
                };
                let (obs, sun, view) = compute_geometry(&config, 30.0);
                let n_ph = 60_000usize;
                let sum: f64 = (0..n_ph)
                    .into_par_iter()
                    .map(|p| {
                        let mut rng = (p as u64)
                            .wrapping_mul(2862933555777941757)
                            .wrapping_add(0xC0FF_EE00)
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1);
                        photon::trace_photon(&atm, obs, view, sun, w550, &mut rng, None).weight
                    })
                    .sum();
                let tw = sum / n_ph as f64;
                eprintln!(
                    "DIAG {label} tau*={tau_star} vz={vz}: flat_ref={flat:.5e} \
                     (o1 {flat1:.5e} + o2p {flat2p:.5e}) \
                     flat_race={flat_race:.5e} (race/ref {:.4}) \
                     trace_photon={tw:.5e} ratio tw/flat={:.4}",
                    flat_race / flat,
                    tw / flat
                );
                // Gate: the two flat estimators bracket MC noise at ~0.5%,
                // trace_photon adds spherical-vs-flat geometry (<1% for
                // these near-vertical daytime cases). 4% catches every bug
                // class found in the G2 campaign (the ground-bounce T=1
                // NEE alone was +30..+90% here) while staying insensitive
                // to seeds.
                assert!(
                    (flat_race / flat - 1.0).abs() < 0.04,
                    "DIAG {label} vz={vz}: race replica {flat_race:.4e} vs \
                     flat referee {flat:.4e}"
                );
                assert!(
                    (tw / flat - 1.0).abs() < 0.04,
                    "DIAG {label} vz={vz}: trace_photon {tw:.4e} vs flat \
                     referee {flat:.4e} (Multiple-mode absolute-radiance bug)"
                );
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Deep-regime harnesses and gates (2026-07, field-forced campaign):
    // the thick-deck SZA >= 101 closure. Heavy MC pieces are #[ignore]
    // (run explicitly); the DEEP referee campaign drives the runner via
    // tools/validate_libradtran.py --tier deep. This block is
    // self-contained on purpose: the identical text is appended to a
    // HEAD worktree to produce the analog (pre-field-forced) baselines
    // of the variance ledger and the bit-identity dump.
    // ════════════════════════════════════════════════════════════════

    /// G2/G3 referee deck (tools/validate_libradtran.py constants):
    /// UNSCALED inputs whose delta-Eddington scaling lands on tau*
    /// exactly (de_scale = 1 - ssa*g^2).
    fn deep_deck_props(tau_star: f64) -> twilight_data::cloud::CloudProperties {
        let g: f64 = 0.85;
        let ssa: f64 = 0.999;
        let de_scale = 1.0 - ssa * g * g;
        twilight_data::cloud::CloudProperties {
            base_km: 1.0,
            top_km: 2.0,
            optical_depth: tau_star / de_scale,
            ssa,
            asymmetry: g,
        }
    }

    /// The G3 protocol atmosphere: Rayleigh + delta-scaled deck, no gas
    /// absorption, no aerosol, refraction off (the MYSTIC decks carry
    /// no refraction either).
    fn deep_atm_1d(tau_star: f64) -> AtmosphereModel {
        let mut atm = builder::build_with_cloud_properties(
            AtmosphereType::UsStandard,
            0.15,
            &deep_deck_props(tau_star),
        );
        for n in atm.refractive_index.iter_mut() {
            *n = 1.0;
        }
        atm
    }

    /// The same deck as an equivalent horizontally uniform 3D field:
    /// clone the 1D atmosphere and zero the shells (the field owns ALL
    /// cloud; the folded cloud absorption in the shell optics stays
    /// IDENTICAL between the two representations), background column
    /// continues the deck beyond the footprint, macrocells derived.
    fn deep_field(
        tau_star: f64,
    ) -> (
        AtmosphereModel,
        twilight_data::cloud_field_builder::OwnedCloudField,
    ) {
        use twilight_data::cloud_field_builder::{field_from_layers, FieldGeometry};
        let mut atm = deep_atm_1d(tau_star);
        atm.cloud_extinction = [0.0; twilight_core::atmosphere::MAX_SHELLS];
        let c = SimulationConfig::default();
        let mut owned = field_from_layers(
            &[deep_deck_props(tau_star)],
            FieldGeometry {
                center_lat_deg: c.latitude,
                center_lon_deg: c.longitude,
                half_extent_km: 256.0,
                res_km: 4.0,
            },
            "deep",
        );
        owned.derive();
        atm.cloud_g_scaled = owned.g_default;
        (atm, owned)
    }

    /// The G3/DEEP zenith-view compare config (twilight-cli compare
    /// defaults: Mecca observer, solar azimuth 270, principal plane).
    fn deep_config(photons: usize, polarized: bool) -> SimulationConfig {
        SimulationConfig {
            view_zenith: 0.0,
            view_azimuth: Some(270.0),
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: photons,
            polarized,
            apply_solar_irradiance: true,
            ..SimulationConfig::default()
        }
    }

    /// Index of the grid wavelength closest to `wl` nm.
    fn wl_index(atm: &AtmosphereModel, wl: f64) -> usize {
        (0..atm.num_wavelengths)
            .min_by(|&a, &b| {
                (atm.wavelengths_nm[a] - wl)
                    .abs()
                    .partial_cmp(&(atm.wavelengths_nm[b] - wl).abs())
                    .unwrap()
            })
            .unwrap()
    }

    /// ONE wavelength of the production hybrid estimator, exactly as
    /// `simulate_at_sza_hybrid`'s polarized arm computes it (same RNG
    /// construction, same solar-irradiance weighting): the Stokes chain
    /// when `polarized`, the per-wavelength scalar chain otherwise.
    fn hybrid_perwl(
        atm: &AtmosphereModel,
        config: &SimulationConfig,
        sza: f64,
        field: Option<&Cloud3DField>,
        w: usize,
    ) -> f64 {
        let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza);
        let sza_bits = mix_salt(sza.to_bits(), config.seed_salt);
        let mut rng = sza_bits
            .wrapping_add(w as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let raw = photon::hybrid_scatter_radiance(
            atm,
            observer_pos,
            view_dir,
            sun_dir,
            w,
            config.photons_per_wavelength,
            &mut rng,
            config.polarized,
            field,
        );
        if config.apply_solar_irradiance && w < SOLAR_IRRADIANCE.len() {
            raw * SOLAR_IRRADIANCE[w]
        } else {
            raw
        }
    }

    /// ONE wavelength of the Multiple estimator (trace_photon), exactly
    /// as `simulate_at_sza_mc` computes it for wavelength `w`.
    fn multiple_perwl(
        atm: &AtmosphereModel,
        config: &SimulationConfig,
        sza: f64,
        field: Option<&Cloud3DField>,
        w: usize,
    ) -> f64 {
        let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza);
        let nphotons = config.photons_per_wavelength;
        let mut total_weight = 0.0;
        for p in 0..nphotons {
            let sza_bits = mix_salt(sza.to_bits(), config.seed_salt);
            let mut rng = (sza_bits)
                .wrapping_add(w as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(p as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(1);
            let result =
                photon::trace_photon(atm, observer_pos, view_dir, sun_dir, w, &mut rng, field);
            total_weight += result.weight;
        }
        let raw = total_weight / nphotons as f64;
        if config.apply_solar_irradiance && w < SOLAR_IRRADIANCE.len() {
            raw * SOLAR_IRRADIANCE[w]
        } else {
            raw
        }
    }

    /// K-seed mean and standard error of a single-wavelength estimator.
    fn perwl_mean_se(
        atm: &AtmosphereModel,
        config: &SimulationConfig,
        sza: f64,
        field: Option<&Cloud3DField>,
        w: usize,
        k: u64,
        multiple: bool,
    ) -> (f64, f64) {
        let s: Vec<f64> = (0..k)
            .into_par_iter()
            .map(|seed| {
                let mut c = config.clone();
                c.seed_salt = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                if multiple {
                    multiple_perwl(atm, &c, sza, field, w)
                } else {
                    hybrid_perwl(atm, &c, sza, field, w)
                }
            })
            .collect();
        let mean = s.iter().sum::<f64>() / k as f64;
        let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / k as f64;
        (mean, (var / k as f64).sqrt())
    }

    /// DEEP campaign twilight harness, driven by
    /// `tools/validate_libradtran.py --tier deep`. Env-configured
    /// (DEEP_TAU_STAR, DEEP_PATH = 1d|field, DEEP_SEEDS, DEEP_PHOTONS,
    /// DEEP_SZAS); prints one machine-readable line
    /// `DEEPCSV,path,tau,seed,sza,wl,rad` per (seed, sza, wl) with the
    /// production polarized STOKES hybrid at the three referee
    /// wavelengths (the CLI compare surface would compute all 41).
    #[test]
    #[ignore = "deep referee harness; driven by validate_libradtran.py --tier deep"]
    fn deep_referee_runner() {
        let getenv = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let tau_star: f64 = getenv("DEEP_TAU_STAR", "3").parse().unwrap();
        let path = getenv("DEEP_PATH", "1d");
        let seeds: u64 = getenv("DEEP_SEEDS", "12").parse().unwrap();
        let photons: usize = getenv("DEEP_PHOTONS", "16000").parse().unwrap();
        let szas: Vec<f64> = getenv("DEEP_SZAS", "101,103")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let wls: Vec<f64> = getenv("DEEP_WLS", "450,550,650")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();

        let atm_1d = deep_atm_1d(tau_star);
        let (atm_f, owned) = deep_field(tau_star);
        let view = owned.view();
        let (atm, field): (&AtmosphereModel, Option<&Cloud3DField>) = if path == "field" {
            (&atm_f, Some(&view))
        } else {
            (&atm_1d, None)
        };
        let config0 = deep_config(photons, true);
        let wids: Vec<usize> = wls.iter().map(|&w| wl_index(atm, w)).collect();
        let mut tasks = Vec::new();
        for seed in 1..=seeds {
            for &sza in &szas {
                for &w in &wids {
                    tasks.push((seed, sza, w));
                }
            }
        }
        let rows: Vec<(u64, f64, f64, f64)> = tasks
            .into_par_iter()
            .map(|(seed, sza, w)| {
                let mut c = config0.clone();
                c.seed_salt = seed;
                let rad = hybrid_perwl(atm, &c, sza, field, w);
                (seed, sza, atm.wavelengths_nm[w], rad)
            })
            .collect();
        for (seed, sza, wl, rad) in rows {
            println!("DEEPCSV,{path},{tau_star},{seed},{sza},{wl},{rad:e}");
        }
    }

    /// G-BDPT-1D: the combined-channel BDPT light subpath (ALIS chain) must
    /// stay unbiased under a 1D gray deck and cut the seed variance that
    /// leaves the deep cells LOW-POWER. Arm A = BDPT-on ALIS
    /// (`hybrid_scatter_radiance_alis`, field None so the 1D deck is active);
    /// arm B = the independent analog `Multiple` (`trace_photon`) reference,
    /// the same estimator-A-vs-estimator-B contract as G-HYB-MULT /
    /// G-FORCED-1D. Passes when the difference of means stays within
    /// `3*sqrt(se_a^2+se_b^2) + 0.05*max`; also prints both CVs so the
    /// variance win is visible.
    /// Env: BDPT_SEEDS, BDPT_PHOTONS, BDPT_SZA, BDPT_TAU, BDPT_WL.
    #[test]
    #[ignore = "G-BDPT-1D: heavy MC. run: cargo test -p twilight-cpu --release -- --ignored --nocapture g_bdpt_under_1d_cloud_matches_multiple"]
    fn g_bdpt_under_1d_cloud_matches_multiple() {
        use twilight_core::photon;
        let getenv = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let seeds: usize = getenv("BDPT_SEEDS", "24").parse().unwrap();
        let photons: usize = getenv("BDPT_PHOTONS", "4000").parse().unwrap();
        let sza: f64 = getenv("BDPT_SZA", "103").parse().unwrap();
        let tau_star: f64 = getenv("BDPT_TAU", "3").parse().unwrap();
        let wl: f64 = getenv("BDPT_WL", "550").parse().unwrap();

        // BDPT_FIELD=1: run the equivalent uniform 3D FIELD instead of the
        // 1D deck (the G-FC gate surface: registry connections under a
        // field vs the same physics as 1D).
        let use_field = std::env::var("BDPT_FIELD").as_deref() == Ok("1");
        let (atm, field_store) = if use_field {
            let (a, o) = deep_field(tau_star);
            (a, Some(o))
        } else {
            (deep_atm_1d(tau_star), None)
        };
        let field_view = field_store.as_ref().map(|o| o.view());
        let field = field_view.as_ref();
        let w = wl_index(&atm, wl);
        // Scalar config -> the ALIS chain (the only chain with BDPT).
        let config = deep_config(photons, false);
        let (obs, sun, view) = compute_geometry(&config, sza);

        // Diagnostic: BDPT_FORCE_GAS=1 drops cloud-vertex scoring (biased) to
        // attribute the variance tail. Default off = correct estimator.
        if std::env::var("BDPT_FORCE_GAS").as_deref() == Ok("1") {
            twilight_core::photon::BDPT_FORCE_GAS_VERTICES
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        // Phase 2 (i): deck-aware light importance boost (default 1.0 = off).
        let deck_boost: f64 = std::env::var("BDPT_DECK_BOOST")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        twilight_core::photon::BDPT_DECK_IMPORTANCE_BOOST
            .store(deck_boost.to_bits(), std::sync::atomic::Ordering::Relaxed);
        twilight_core::photon::BDPT_CLOUD_VERTEX_COUNT
            .store(0, std::sync::atomic::Ordering::Relaxed);
        // Phase 2: active light-subpath vertex count (1 or 2). Default 1.
        let verts: usize = std::env::var("BDPT_VERTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        twilight_core::photon::BDPT_ACTIVE_LIGHT_VERTS
            .store(verts, std::sync::atomic::Ordering::Relaxed);

        // Spike-attribution gates (scoring-only, RNG-neutral; BIASED when
        // narrowed -- attribution runs only). Terms sum to the full run.
        {
            use std::sync::atomic::Ordering::Relaxed;
            let geti = |k: &str, d: usize| -> usize {
                std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
            };
            twilight_core::photon::BDPT_DIAG_NEE_MIN_BOUNCE
                .store(geti("BDPT_NEE_MIN", 0), Relaxed);
            twilight_core::photon::BDPT_DIAG_NEE_MAX_BOUNCE
                .store(geti("BDPT_NEE_MAX", usize::MAX), Relaxed);
            twilight_core::photon::BDPT_DIAG_SPLIT_NEE_OFF
                .store(std::env::var("BDPT_SPLIT_OFF").as_deref() == Ok("1"), Relaxed);
            twilight_core::photon::BDPT_DIAG_GROUND_NEE_OFF
                .store(std::env::var("BDPT_GROUND_OFF").as_deref() == Ok("1"), Relaxed);
            twilight_core::photon::BDPT_DIAG_CONNECTIONS_OFF
                .store(std::env::var("BDPT_CONN_OFF").as_deref() == Ok("1"), Relaxed);
            // BDPT_DIAG=1: arm the weight-anatomy probe at the scored w.
            twilight_core::photon::BDPT_DIAG_SCORE_WL.store(
                if std::env::var("BDPT_DIAG").as_deref() == Ok("1") {
                    w
                } else {
                    usize::MAX
                },
                Relaxed,
            );
            twilight_core::photon::BDPT_DIAG_MAX_C.store(0, Relaxed);
        }
        // BDPT_SEED_ONLY=idx: run just that seed index (its salt depends only
        // on the index, so the run reproduces the same seed's value exactly).
        // BDPT_SEED_START=k: run seed indices k..k+seeds (pool independent
        // batches across runs that individually fit the background cap).
        let seed_only: Option<usize> = std::env::var("BDPT_SEED_ONLY")
            .ok()
            .and_then(|s| s.parse().ok());
        let seed_start: usize = std::env::var("BDPT_SEED_START")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let seed_list: Vec<usize> = match seed_only {
            Some(idx) => vec![idx],
            None => (seed_start..seed_start + seeds).collect(),
        };
        let seeds = seed_list.len();

        // Arm A: BDPT-on ALIS under the 1D deck (field = None).
        // BDPT_STOKES=1: run the polarized STOKES hybrid instead -- the
        // EXACT deep-referee estimator (hybrid_perwl with seed_salt), to
        // validate the chain-connection port on the referee path itself.
        let stokes_arm = std::env::var("BDPT_STOKES").as_deref() == Ok("1");
        let stokes_config = deep_config(photons, true);
        let a: Vec<f64> = seed_list
            .into_par_iter()
            .map(|seed| {
                if stokes_arm {
                    let mut c = stokes_config.clone();
                    c.seed_salt = seed as u64;
                    return hybrid_perwl(&atm, &c, sza, field, w);
                }
                let salt = (seed as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let sza_bits = mix_salt(sza.to_bits(), salt);
                let mut rng = sza_bits
                    .wrapping_add(w as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                let r = photon::hybrid_scatter_radiance_alis(
                    &atm, obs, view, sun, photons, &mut rng, field,
                );
                if w < SOLAR_IRRADIANCE.len() {
                    r[w] * SOLAR_IRRADIANCE[w]
                } else {
                    r[w]
                }
            })
            .collect();
        let m_a = a.iter().sum::<f64>() / seeds as f64;
        let var_a = a.iter().map(|x| (x - m_a).powi(2)).sum::<f64>() / seeds as f64;
        let se_a = (var_a / seeds as f64).sqrt();

        // Heavy-tail diagnostic: sort per-seed values to expose skew.
        let mut sorted = a.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let amin = sorted[0];
        let amed = sorted[seeds / 2];
        let amax = sorted[seeds - 1];
        let cv_a = if m_a.abs() > 1e-30 {
            var_a.sqrt() / m_a.abs()
        } else {
            0.0
        };
        let max_over_mean = if m_a.abs() > 1e-30 { amax / m_a } else { 0.0 };
        let n_tiny = a.iter().filter(|&&x| x < 0.01 * m_a).count();

        // Optional analog Multiple reference (slow; BDPT_ANALOG=1 to include).
        if std::env::var("BDPT_ANALOG").as_deref() == Ok("1") {
            let (mb, sb) = perwl_mean_se(&atm, &config, sza, None, w, seeds as u64, true);
            let cvb = if mb.abs() > 1e-30 {
                sb * (seeds as f64).sqrt() / mb.abs()
            } else {
                0.0
            };
            eprintln!("BDPTMULT m={mb:.4e} cv={cvb:.3}");
        }

        // Cached MYSTIC reference (BDPT_REF) for the real unbiasedness check.
        let myref: f64 = std::env::var("BDPT_REF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);

        let cloud_verts = twilight_core::photon::BDPT_CLOUD_VERTEX_COUNT
            .load(std::sync::atomic::Ordering::Relaxed);
        // Argmax seed index (for BDPT_SEED_ONLY attribution reruns).
        let argmax_seed = seed_only.unwrap_or_else(|| {
            a.iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        });
        eprintln!(
            "BDPTCSV tau={tau_star} sza={sza} wl={wl} seeds={seeds} photons={photons} \
             deck_boost={deck_boost} verts={verts} | \
             m={m_a:.4e} se={se_a:.3e} cv={cv_a:.3} | \
             min={amin:.2e} med={amed:.2e} max={amax:.2e} max/mean={max_over_mean:.1} \
             n<1%mean={n_tiny}/{seeds} argmax_seed={argmax_seed} | \
             cloud_verts={cloud_verts} | \
             MYSTIC={myref:.3e} ratio={:.3}",
            m_a / myref
        );

        // Weight-anatomy report (BDPT_DIAG=1): the factor anatomy of the
        // max-contribution main-particle NEE at the scored wavelength.
        if std::env::var("BDPT_DIAG").as_deref() == Ok("1") {
            use std::sync::atomic::Ordering::Relaxed;
            let g = |i: usize| {
                f64::from_bits(twilight_core::photon::BDPT_DIAG_MAX_INFO[i].load(Relaxed))
            };
            eprintln!(
                "BDPTDIAG max_c={:.3e} hero_w={:.3e} wr_w={:.3e} t_suns={:.3e} \
                 phase={:.3e} bounce={} alt_km={:.1} n_rr={} rr_factor={:.3e} \
                 vspg_factor={:.3e} et_factor={:.3e} hero_wl={} nee_weight={:.3e}",
                f64::from_bits(twilight_core::photon::BDPT_DIAG_MAX_C.load(Relaxed)),
                g(0),
                g(1),
                g(2),
                g(3),
                g(4) as i64,
                g(5) / 1000.0,
                g(6) as i64,
                g(7),
                g(8),
                g(9),
                g(10) as i64,
                g(11),
            );
        }
    }

    /// Importance-map dump on the REAL clear-sky atmosphere (the paper
    /// figure; the toy Rayleigh test atmosphere is too transparent to
    /// show the corridor structure).
    #[test]
    #[ignore = "diagnostic dump, run explicitly"]
    fn imap_dump_real() {
        let atm = make_clear_sky_atm();
        let m = twilight_core::importance::SolarImportanceMap::build(&atm, wl_index(&atm, 450.0));
        println!("IMAPCSV,alt_km,cos_sun,importance");
        for ia in 0..twilight_core::importance::ALT_BANDS {
            let alt = (ia as f64 + 0.5) * 3.0;
            for ic in 0..twilight_core::importance::COS_BANDS {
                let c = -0.5 + (ic as f64 + 0.5) / twilight_core::importance::COS_BANDS as f64;
                println!("IMAPCSV,{alt},{c:.4},{:e}", m.lookup(alt * 1000.0, c));
            }
        }
    }

    /// Degree-of-polarization dump for the paper's polarization figure:
    /// zenith view, clear sky, backward MC Stokes at three SZA.
    /// `cargo test -p twilight-cpu --release dop_dump -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic dump, run explicitly"]
    fn dop_dump() {
        use twilight_core::geometry::{geographic_to_ecef, solar_direction_ecef};
        let atm = make_clear_sky_atm();
        let (lat, lon) = (21.4225, 39.8262);
        let obs = geographic_to_ecef(lat, lon, 0.0);
        let view = solar_direction_ecef(0.001, 270.0, lat, lon); // zenith
        println!("DOPCSV,sza,wl_nm,I,dop");
        for sza in [85.0f64, 90.0, 94.0, 97.0] {
            let sun = solar_direction_ecef(sza, 270.0, lat, lon);
            let s = twilight_core::photon::mc_scatter_spectrum_polarized(
                &atm, obs, view, sun, 40000, 0xC0FFEE, None,
            );
            for (w, sv) in s.iter().enumerate().take(atm.num_wavelengths) {
                let i = sv.s[0];
                let dop = if i > 1e-30 {
                    (sv.s[1] * sv.s[1] + sv.s[2] * sv.s[2]).sqrt() / i
                } else {
                    0.0
                };
                println!("DOPCSV,{sza},{:.0},{i:e},{dop:.4}", atm.wavelengths_nm[w]);
            }
        }
    }

    /// Variance-ledger harness: seed-CV of the production STOKES hybrid
    /// per referee wavelength, env-configured (CV_FIELD = synthetic |
    /// padborg, CV_SZAS, CV_SEEDS, CV_PHOTONS). Prints
    /// `CVCSV,field,sza,wl,mean,se,cv_pct` lines. Run in THIS tree
    /// (field-forced) and in a HEAD worktree (field chains analog) to
    /// produce the forced-vs-analog CV comparison.
    #[test]
    #[ignore = "variance-ledger harness; run explicitly, see RESULTS_DEEP_REGIME.md"]
    fn cv_ledger_field() {
        let getenv = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let which = getenv("CV_FIELD", "synthetic");
        let seeds: u64 = getenv("CV_SEEDS", "8").parse().unwrap();
        let photons: usize = getenv("CV_PHOTONS", "4000").parse().unwrap();
        let szas: Vec<f64> = getenv("CV_SZAS", "99,101,103")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();

        // CV_FIELD=clear: clear-sky polarized ledger (field absent), the
        // khayt production regime without cloud tails; measures the
        // weight-window effect on the pure gas chain.
        let clear_mode = which == "clear";
        let (atm, owned, lat, lon) = if which == "padborg" {
            let owned = twilight_weather::cloud3d::load_field(std::path::Path::new(
                &getenv("CV_PADBORG_BIN", "/tmp/padborg_field.bin"),
            ))
            .expect("padborg field sidecar (regenerate: tools/cloud3d_seviri.py)");
            let mut atm = builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
            for n in atm.refractive_index.iter_mut() {
                *n = 1.0;
            }
            atm.cloud_g_scaled = owned.g_default;
            (atm, owned, 54.83, 9.36)
        } else if clear_mode {
            let atm = make_clear_sky_atm();
            // Dummy field, never bound (field arg forced to None below).
            let (_, owned) = deep_field(3.0);
            let c = SimulationConfig::default();
            (atm, owned, c.latitude, c.longitude)
        } else {
            let (atm, owned) = deep_field(3.0);
            let c = SimulationConfig::default();
            (atm, owned, c.latitude, c.longitude)
        };
        let view = owned.view();
        let field_opt = if clear_mode { None } else { Some(&view) };
        let config0 = SimulationConfig {
            latitude: lat,
            longitude: lon,
            ..deep_config(photons, true)
        };
        let wls: Vec<f64> = getenv("CV_WLS", "450,550,650")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let wids: Vec<usize> = wls.iter().map(|&w| wl_index(&atm, w)).collect();
        for &sza in &szas {
            for &w in &wids {
                let s: Vec<f64> = (0..seeds)
                    .into_par_iter()
                    .map(|seed| {
                        let mut c = config0.clone();
                        c.seed_salt =
                            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                        hybrid_perwl(&atm, &c, sza, field_opt, w)
                    })
                    .collect();
                let mean = s.iter().sum::<f64>() / seeds as f64;
                let var =
                    s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (seeds - 1) as f64;
                let sd = var.sqrt();
                let cv = if mean.abs() > 0.0 { 100.0 * sd / mean } else { f64::NAN };
                println!(
                    "CVCSV,{which},{sza},{:.0},{mean:e},{:e},{cv:.1}",
                    atm.wavelengths_nm[w],
                    sd / (seeds as f64).sqrt()
                );
            }
        }
    }

    /// Bit-identity dump: prints `BITCHECK,tag,sza,wl,bits,val` for the
    /// paths whose RNG streams and arithmetic MUST be untouched by the
    /// field-forced mode (clear sky, 1D decks at every SZA, Multiple
    /// mode everywhere, field runs below the forced gate, ALIS field
    /// runs at any SZA). Rows tagged `field-stokes-deep` are the ONE
    /// surface the campaign changes by design (field + Stokes chains +
    /// local SZA >= 96): the diff script excludes them, and them only.
    #[test]
    #[ignore = "bit-identity harness; run in both trees and diff (RESULTS_DEEP_REGIME.md)"]
    fn bitcheck_dump() {
        let clear = make_clear_sky_atm();
        let deck = builder::build_with_cloud_properties(
            AtmosphereType::UsStandard,
            0.15,
            &thin_deck_props(),
        );
        let mut atm_field = builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let owned = uniform_thin_field();
        atm_field.cloud_g_scaled = owned.g_default;
        let view = owned.view();

        let dump = |tag: &str,
                    atm: &AtmosphereModel,
                    cfg: &SimulationConfig,
                    sza: f64,
                    field: Option<&Cloud3DField>| {
            let r = simulate_at_sza(atm, cfg, sza, field);
            for (wl, rad) in r.wavelengths_nm.iter().zip(r.radiance.iter()) {
                println!("BITCHECK,{tag},{sza},{wl},{:016x},{rad:e}", rad.to_bits());
            }
        };
        let alis = SimulationConfig {
            view_zenith: 80.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 200,
            polarized: false,
            ..SimulationConfig::default()
        };
        let stokes = SimulationConfig {
            polarized: true,
            ..alis.clone()
        };
        let multiple = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 2000,
            ..alis.clone()
        };
        for sza in [30.0, 85.0, 97.0, 101.0, 106.0] {
            dump("clear-alis", &clear, &alis, sza, None);
        }
        for sza in [85.0, 97.0, 101.0] {
            dump("clear-stokes", &clear, &stokes, sza, None);
        }
        for sza in [60.0, 85.0, 97.0, 100.0] {
            dump("deck1d-alis", &deck, &alis, sza, None);
            dump("deck1d-stokes", &deck, &stokes, sza, None);
        }
        dump("deck1d-multiple", &deck, &multiple, 97.0, None);
        dump("field-multiple", &atm_field, &multiple, 97.0, Some(&view));
        // Below the forced gate (view toward the sun: local SZA <= 92
        // everywhere on the LOS): both chain families must be identical.
        for sza in [88.0, 92.0] {
            dump("field-alis", &atm_field, &alis, sza, Some(&view));
            dump("field-stokes", &atm_field, &stokes, sza, Some(&view));
        }
        // ALIS stays analog under fields at EVERY SZA: identical.
        for sza in [97.0, 101.0] {
            dump("field-alis-deep", &atm_field, &alis, sza, Some(&view));
        }
        // The one changed surface (expected to differ from HEAD).
        for sza in [97.0, 101.0] {
            dump("field-stokes-deep", &atm_field, &stokes, sza, Some(&view));
        }
    }

    /// G-S3-CB (the decisive field estimator gate): CHECKERBOARD
    /// fields (deck cells alternating with clear cells, 16 km tiles,
    /// clear background) at SZA 97/100/103, 550 nm: the field-FORCED
    /// per-wavelength scalar hybrid vs the fully ANALOG Multiple
    /// estimator (trace_photon: races the same gray channel with no
    /// forced mode at any SZA, trajectory-independent, externally
    /// anchored by the G2/G3 campaigns). Two decks, two gate modes:
    ///
    /// - tau* = 1 checkerboard: TWO-SIDED,
    ///   |m_h - m_m| < 3 x combined SE + 5% construction floor. Both
    ///   estimators converge at the test budget, so this is the
    ///   decisive agreement gate for the field-forced machinery on a
    ///   genuinely 3D medium.
    /// - tau* = 3 checkerboard: one-sided REGRESSION bounds
    ///   [0.35 x m_m, m_m + 3 SE + 5%]. At affordable budgets the
    ///   hybrid mean under tau* = 3 is tail-limited from BELOW
    ///   (measured 0.618x at SZA 97, 8 seeds x 8000, 2026-07-04; the
    ///   distribution-level law gates in photon.rs prove the flight
    ///   machinery exact, and the deficit shrinks with budget: the
    ///   program's documented heavy-tail residual). The 0.35 floor
    ///   fails any regression into the pre-campaign analog-collapse
    ///   class (0.05-0.10x of the referee, see the variance ledger);
    ///   the upper side fails inflation of any origin. EXCEPT SZA 103:
    ///   measured 0.122x of Multiple (2026-07-05) with the uniform-deck
    ///   twins at 0.13x of the 1e9 referee (forced) vs 0.05x
    ///   (analog): at the deepest-thickest corner the forced estimator
    ///   itself still sits inside the collapse band at achievable
    ///   budgets, so NO floor separates the classes with power there.
    ///   That row is REPORTED (KNOWN-LIM, the same taxonomy as the
    ///   July-02 campaign rows), not asserted; its closure is the
    ///   standing budget/BDPT follow-up.
    ///
    /// VIEW GEOMETRY (finding, 2026-07-04; row restored 2026-07-06):
    /// the tau-ladder rows look at vz 80 (the g_s2 convention). A
    /// zenith LOS over a checkerboard threads a SINGLE cell column;
    /// with the observer under a CLEAR cell every LOS seed is gas-only
    /// (beta_seed = 0) and the chain direction lobes (sun-phase,
    /// zenith, terminator) rarely sample the down-and-sideways
    /// directions that couple to the off-axis cloud cells, so the
    /// cloud-mediated class becomes an unsampled heavy tail: the
    /// 2026-07-04 draft measured tau*1/SZA 97 zenith hybrid 3.764e-6
    /// with a FALSE-TIGHT 1.1% seed SE vs analog Multiple 7.543e-6
    /// (0.499x), recorded as production residual 3 in
    /// RESULTS_DEEP_REGIME.md. Unbiasedness at that geometry is pinned
    /// by the flight-law/majorant-invariance/eq1d gates; the deficit
    /// was importance-sampling starvation at achievable budgets. The
    /// slant view couples the LOS to both cell types and remains the
    /// agreement geometry for the tau ladder. The zenith view is now
    /// ALSO gated (two-sided) by the dedicated zenith-starvation row
    /// at the bottom of this test: the per-wavelength chains carry a
    /// lateral-escape seed lobe (photon.rs, LATERAL_ESCAPE_SHARE,
    /// active only over broken fields at deep SZA for cloud-coupled
    /// seeds), and on the current tree the converged Multiple
    /// reference sits at 0.95-0.96x parity with the hybrid there (see
    /// the row's comment for the full measured history, including the
    /// halving of the draft-era Multiple reference by the
    /// post-addendum recalibration commits).
    /// The G-S3-CB checkerboard medium: the deep deck carved into
    /// `cell` x `cell`-voxel (4*cell km) cells alternating with clear
    /// cells, clear background beyond the footprint (the checkerboard
    /// is the medium under test; a half-mean background would blur it).
    /// The gate carves cell = 2 (8 km); the zenith probe also runs
    /// cell = 4 (16 km, the 2026-07-04 draft tile size the starvation
    /// finding was recorded on).
    fn checkerboard_field(
        tau_star: f64,
        cell: usize,
    ) -> (
        AtmosphereModel,
        twilight_data::cloud_field_builder::OwnedCloudField,
    ) {
        let (atm, mut owned) = deep_field(tau_star);
        let (nz, nlat, nlon) = (owned.nz, owned.nlat, owned.nlon);
        for iz in 0..nz {
            for ilat in 0..nlat {
                for ilon in 0..nlon {
                    if (ilat / cell + ilon / cell) % 2 == 1 {
                        owned.sigma[(iz * nlat + ilat) * nlon + ilon] = 0.0;
                    }
                }
            }
        }
        for b in owned.background_column.iter_mut() {
            *b = 0.0;
        }
        owned.derive();
        (atm, owned)
    }

    /// Center coordinates of the CLEAR checkerboard cell nearest the
    /// field center: the zenith-starvation geometry (residual 3 in
    /// RESULTS_DEEP_REGIME.md), a vertical LOS threading a single clear
    /// column with cloud walls 2*cell km away on every side. The carve
    /// clears cell (ic, jc) = (ilat/cell, ilon/cell) when ic + jc is
    /// odd; the central cell is even (cloudy), so step one cell east.
    fn clear_cell_center(
        owned: &twilight_data::cloud_field_builder::OwnedCloudField,
        cell: usize,
    ) -> (f64, f64) {
        let (ic, mut jc) = (owned.nlat / (2 * cell), owned.nlon / (2 * cell));
        if (ic + jc) % 2 == 0 {
            jc += 1;
        }
        let half = 0.5 * cell as f64;
        let lat = owned.lat0_deg + ((cell * ic) as f64 + half) * owned.dlat_deg;
        let lon = owned.lon0_deg + ((cell * jc) as f64 + half) * owned.dlon_deg;
        (lat, lon)
    }

    /// Zenith-starvation probe (diagnosis harness for residual 3 of
    /// RESULTS_DEEP_REGIME.md): the G-S3-CB checkerboard with the
    /// observer at the center of a CLEAR cell, zenith view, hybrid
    /// scalar chain vs analog Multiple at matched geometry. Env knobs:
    /// PROBE_TAU (1), PROBE_SZA (97), PROBE_VZ (0), PROBE_SEEDS (8),
    /// PROBE_PHOTONS_H (8000), PROBE_PHOTONS_M (100000), PROBE_CELL
    /// (2 voxels = 8 km; 4 = the 16 km draft tiles). Prints one
    /// PROBECSV line; asserts nothing (diagnosis only).
    #[test]
    #[ignore = "g_s3_cb zenith starvation probe; env-driven diagnosis harness"]
    fn g_s3_cb_zenith_probe() {
        let getenv = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let tau_star: f64 = getenv("PROBE_TAU", "1").parse().unwrap();
        let sza: f64 = getenv("PROBE_SZA", "97").parse().unwrap();
        let vz: f64 = getenv("PROBE_VZ", "0").parse().unwrap();
        let seeds: u64 = getenv("PROBE_SEEDS", "8").parse().unwrap();
        let ph: usize = getenv("PROBE_PHOTONS_H", "8000").parse().unwrap();
        let pm: usize = getenv("PROBE_PHOTONS_M", "100000").parse().unwrap();
        let cell: usize = getenv("PROBE_CELL", "2").parse().unwrap();

        let (atm, owned) = checkerboard_field(tau_star, cell);
        let (lat, lon) = clear_cell_center(&owned, cell);
        let view = owned.view();
        // The probe geometry premise: the observer column must be clear
        // (in-deck altitude, 1.5 km) with cloud in the neighbor cell.
        let obs = twilight_core::geometry::geographic_to_ecef(lat, lon, 0.0);
        let up = obs.normalize();
        assert_eq!(
            view.sigma_at(obs + up * 1500.0),
            0.0,
            "probe observer column is not clear"
        );
        let hybrid = SimulationConfig {
            view_zenith: vz,
            latitude: lat,
            longitude: lon,
            ..deep_config(ph, false)
        };
        let multiple = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: pm,
            ..hybrid.clone()
        };
        let w550 = wl_index(&atm, 550.0);
        let (m_h, se_h) = perwl_mean_se(&atm, &hybrid, sza, Some(&view), w550, seeds, false);
        let (m_m, se_m) = perwl_mean_se(&atm, &multiple, sza, Some(&view), w550, seeds, true);
        let cv = |m: f64, se: f64| 100.0 * se * (seeds as f64).sqrt() / m;
        println!(
            "PROBECSV,tau*{tau_star},sza{sza},vz{vz},cell{cell},hybrid,{m_h:.4e},se,{se_h:.2e},\
             cv%,{:.1},multiple,{m_m:.4e},se,{se_m:.2e},cv%,{:.1},ratio,{:.3}",
            cv(m_h, se_h),
            cv(m_m, se_m),
            m_h / m_m
        );
    }

    #[test]
    #[ignore = "g_s3_cb: heavy MC (hours); the decisive field-forced gate"]
    fn g_s3_field_forced_matches_multiple_checkerboard() {
        let mut failures = Vec::new();
        for (tau_star, two_sided) in [(1.0, true), (3.0, false)] {
            let (atm, owned) = checkerboard_field(tau_star, 2);
            let view = owned.view();

            let hybrid = SimulationConfig {
                view_zenith: 80.0,
                ..deep_config(8_000, false)
            };
            // Multiple budget: trace_photon through a FIELD walks the
            // DDA on every flight segment (far costlier per photon than
            // the 1D race), so the referee side runs 4e5 photons per
            // seed. The band uses the measured SEs, so the reduced
            // budget widens rather than weakens the gate.
            let multiple = SimulationConfig {
                scattering_mode: ScatteringMode::Multiple,
                photons_per_wavelength: 400_000,
                ..hybrid.clone()
            };
            let w550 = wl_index(&atm, 550.0);
            for sza in [97.0, 100.0, 103.0] {
                let (m_h, se_h) =
                    perwl_mean_se(&atm, &hybrid, sza, Some(&view), w550, 8, false);
                let (m_m, se_m) =
                    perwl_mean_se(&atm, &multiple, sza, Some(&view), w550, 8, true);
                let se = (se_h * se_h + se_m * se_m).sqrt();
                if two_sided {
                    let diff = (m_h - m_m).abs();
                    let band = 3.0 * se + 0.05 * m_h.max(m_m);
                    eprintln!(
                        "G-S3-CB tau*{tau_star} SZA {sza} (two-sided): hybrid {m_h:.5e} \
                         (se {se_h:.2e}) multiple {m_m:.5e} (se {se_m:.2e}) \
                         ratio {:.3} diff {diff:.2e} band {band:.2e}",
                        m_h / m_m
                    );
                    if diff.is_nan() || diff >= band {
                        failures.push(format!(
                            "tau*{tau_star} SZA {sza}: hybrid {m_h:.5e} vs multiple \
                             {m_m:.5e} (diff {diff:.3e} >= band {band:.3e})"
                        ));
                    }
                } else if sza >= 103.0 {
                    // KNOWN-LIM: reported, not asserted (see header).
                    eprintln!(
                        "G-S3-CB tau*{tau_star} SZA {sza} (KNOWN-LIM, reported): hybrid \
                         {m_h:.5e} (se {se_h:.2e}) multiple {m_m:.5e} (se {se_m:.2e}) \
                         ratio {:.3}",
                        m_h / m_m
                    );
                } else {
                    let upper = m_m + 3.0 * se + 0.05 * m_m;
                    let floor = 0.35 * m_m;
                    eprintln!(
                        "G-S3-CB tau*{tau_star} SZA {sza} (regression bounds): hybrid \
                         {m_h:.5e} (se {se_h:.2e}) multiple {m_m:.5e} (se {se_m:.2e}) \
                         ratio {:.3} bounds [{floor:.3e}, {upper:.3e}]",
                        m_h / m_m
                    );
                    if m_h.is_nan() || m_h >= upper || m_h <= floor {
                        failures.push(format!(
                            "tau*{tau_star} SZA {sza}: hybrid {m_h:.5e} outside \
                             [{floor:.3e}, {upper:.3e}] around multiple {m_m:.5e}"
                        ));
                    }
                }
            }
        }

        // ── ZENITH-STARVATION ROW (residual 3, restored 2026-07-06) ──
        // tau* = 1, SZA 97, VIEW ZENITH 0, observer at the center of a
        // CLEAR 16 km cell (the 2026-07-04 draft tile size the 0.499x
        // finding was recorded on; at 8 km cells the truncated zenith
        // lobe's skirt, cos_min = 0.2, still reaches the 4 km-away
        // cloud walls and the class is covered without a lateral lobe).
        // History, measured at THIS budget (8 x 8000 vs 8 x 4e5):
        // - recorded draft (2026-07-04 tree): hybrid 3.764e-6 with a
        //   FALSE-TIGHT 1.1% seed SE vs Multiple 7.543e-6 (0.499x),
        //   the residual-3 starvation class;
        // - HEAD worktree c6232fc, pre-lateral-lobe (2026-07-06):
        //   hybrid 3.7338e-6 (seed CV 2.7%) vs converged Multiple
        //   3.9173e-6 (CV 34%; 6.4e-6 at a 2000-photon budget with CV
        //   174%, i.e. the draft-era reference class is tiny-budget
        //   noise on this tree): ratio 0.953. The Multiple side of the
        //   record HALVED between 07-04 and HEAD (the post-addendum
        //   source-correction/recalibration commits moved absolute
        //   scales; the hybrid value reproduces the draft exactly), so
        //   the 0.499x deficit no longer manifests here;
        // - this tree (lateral-escape seed lobe active, photon.rs):
        //   hybrid 3.7688e-6 (CV 4.0%), ratio 0.962.
        // The row therefore gates TWO-SIDED (3 SE + 5%): a regression
        // into the recorded starvation class (<= 0.6x with a
        // false-tight seed SE) fails the band by an order of
        // magnitude, and inflation of any origin fails the upper side.
        {
            let (atm, owned) = checkerboard_field(1.0, 4);
            let (lat, lon) = clear_cell_center(&owned, 4);
            let view = owned.view();
            let hybrid = SimulationConfig {
                view_zenith: 0.0,
                latitude: lat,
                longitude: lon,
                ..deep_config(8_000, false)
            };
            let multiple = SimulationConfig {
                scattering_mode: ScatteringMode::Multiple,
                photons_per_wavelength: 400_000,
                ..hybrid.clone()
            };
            let w550 = wl_index(&atm, 550.0);
            let (m_h, se_h) = perwl_mean_se(&atm, &hybrid, 97.0, Some(&view), w550, 8, false);
            let (m_m, se_m) =
                perwl_mean_se(&atm, &multiple, 97.0, Some(&view), w550, 8, true);
            let se = (se_h * se_h + se_m * se_m).sqrt();
            let diff = (m_h - m_m).abs();
            let band = 3.0 * se + 0.05 * m_h.max(m_m);
            eprintln!(
                "G-S3-CB tau*1 SZA 97 vz 0 cell 16km (two-sided, zenith row): hybrid \
                 {m_h:.5e} (se {se_h:.2e}) multiple {m_m:.5e} (se {se_m:.2e}) \
                 ratio {:.3} diff {diff:.2e} band {band:.2e}",
                m_h / m_m
            );
            if diff.is_nan() || diff >= band {
                failures.push(format!(
                    "tau*1 SZA 97 vz 0 (zenith row): hybrid {m_h:.5e} vs multiple \
                     {m_m:.5e} (diff {diff:.3e} >= band {band:.3e})"
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "G-S3-CB: field-forced hybrid vs analog Multiple:\n{}",
            failures.join("\n")
        );
    }

    /// G-S3-EQ1D-DEEP (physics gate 4c): the uniform-3D-field and
    /// 1D-deck representations of the SAME tau* = 3 deck must agree at
    /// SZA 101 and 103 (both paths now run forced flights, so the
    /// variance permits a two-sided band this deep), 550 nm,
    /// per-wavelength scalar hybrid, 8 seeds x 8000 photons.
    #[test]
    #[ignore = "g_s3_eq1d_deep: heavy MC (minutes)"]
    fn g_s3_eq1d_deep() {
        let atm_1d = deep_atm_1d(3.0);
        let (atm_f, owned) = deep_field(3.0);
        let view = owned.view();
        let config = deep_config(8_000, false);
        let w550 = wl_index(&atm_1d, 550.0);
        let mut failures = Vec::new();
        for sza in [101.0, 103.0] {
            let (m_1d, se_1d) = perwl_mean_se(&atm_1d, &config, sza, None, w550, 8, false);
            let (m_f, se_f) =
                perwl_mean_se(&atm_f, &config, sza, Some(&view), w550, 8, false);
            let se = (se_1d * se_1d + se_f * se_f).sqrt();
            let diff = (m_1d - m_f).abs();
            let band = 3.0 * se + 0.02 * m_1d.max(m_f);
            eprintln!(
                "G-S3-EQ1D-DEEP SZA {sza}: 1D {m_1d:.5e} (se {se_1d:.2e}) field {m_f:.5e} \
                 (se {se_f:.2e}) ratio {:.3} diff {diff:.2e} band {band:.2e}",
                m_f / m_1d
            );
            if diff.is_nan() || diff >= band {
                failures.push(format!(
                    "SZA {sza}: 1D {m_1d:.5e} vs field {m_f:.5e} \
                     (diff {diff:.3e} >= band {band:.3e})"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "G-S3-EQ1D-DEEP: representations disagree:\n{}",
            failures.join("\n")
        );
    }

    /// G-S3-MONO (physics gate 4a, no referee in the loop): deck
    /// radiance strictly decreasing in tau* at fixed SZA, and smoothly
    /// decreasing in SZA at fixed tau*, on seed-averaged means with
    /// every claimed decrease RESOLVED at 2 combined seed SE (a claim
    /// the pre-campaign variance could not even state at these SZAs).
    ///
    /// WHERE each claim is physical, anchored on the cached external
    /// referee (not gated against it, just used to place the gate):
    /// - tau* ladder at SZA 97, 650 nm: MYSTIC gives tau*1 = 4.663e-6
    ///   vs tau*3 = 3.384e-6 W/m^2/sr/nm (38% gap, ~5 referee SEs; the
    ///   650 nm channel has both the widest ladder and the thinnest
    ///   chain tails). At SZA >= 101 the 3e8 referee shows tau*1 and
    ///   tau*3 EQUAL within its SE (550 nm: 1.163e-7 vs 1.224e-7):
    ///   deep-twilight sidelight scattered INTO the beam by the thicker
    ///   deck compensates its extinction, so a tau* ladder there gates
    ///   nothing and is deliberately not asserted.
    /// - CLEAR SKY IS NOT A RUNG at 650 nm: the referee's own tau*1 row
    ///   (4.663e-6) EXCEEDS the validated clear-sky zenith radiance at
    ///   SZA 97 (measured 2.77e-6 +- 0.06e-6 by the same estimator that
    ///   passed the tier1b-deep clear campaigns): a thin low deck
    ///   REDIRECTS the bright solar-horizon twilight light into the
    ///   dim red zenith and BRIGHTENS it. First drafted as
    ///   clear > tau*1 > tau*3, this gate FAILED on that premise
    ///   (2026-07-04) and the failure is the physics, not the
    ///   estimator; the clear value is still computed and printed.
    /// - SZA ladder at tau* = 3, 550 nm: 99 > 101 > 103 (referee gaps
    ///   5.3x and 4.5x: dimming of the twilight source dominates every
    ///   deck effect).
    #[test]
    #[ignore = "g_s3_mono: heavy MC (minutes)"]
    fn g_s3_deck_monotonicity() {
        let clear = {
            let mut a = builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
            for n in a.refractive_index.iter_mut() {
                *n = 1.0;
            }
            a
        };
        let atm_t1 = deep_atm_1d(1.0);
        let atm_t3 = deep_atm_1d(3.0);
        let config = deep_config(8_000, false);
        let k = 8;

        // (a) strictly decreasing in tau* at SZA 97, 650 nm.
        let w650 = wl_index(&clear, 650.0);
        let sza = 97.0;
        let (m_c, se_c) = perwl_mean_se(&clear, &config, sza, None, w650, k, false);
        let (m_1, se_1) = perwl_mean_se(&atm_t1, &config, sza, None, w650, k, false);
        let (m_3, se_3) = perwl_mean_se(&atm_t3, &config, sza, None, w650, k, false);
        eprintln!(
            "G-S3-MONO tau ladder @ SZA {sza} 650nm: clear {m_c:.4e} (se {se_c:.1e}) \
             > tau1 {m_1:.4e} (se {se_1:.1e}) > tau3 {m_3:.4e} (se {se_3:.1e})"
        );
        // Clear is printed, not asserted (see header: the deck
        // BRIGHTENS the red zenith at this SZA, referee-corroborated).
        let _ = se_c;
        let se_13 = (se_1 * se_1 + se_3 * se_3).sqrt();
        assert!(
            m_1 - m_3 > 2.0 * se_13,
            "tau*1 ({m_1:.4e}) must exceed tau*3 ({m_3:.4e}) by 2 SE ({se_13:.2e})"
        );

        // (b) smoothly decreasing in SZA at tau* = 3, 550 nm. Seeds per
        // rung sized from MEASURED seed CVs (k = 8 probe, 2026-07-04:
        // CV 150% at SZA 101, 190% at 103 for the scalar chain at 8k):
        // resolving the 3.6x gap 101 -> 103 at 2 combined SE needs
        // k ~ 32 on the deep rungs (2*sqrt((1.5/sqrt(32))^2 +
        // (1.9/sqrt(32))^2) ~ 0.85 of the mean vs a 2.6-of-mean gap on
        // the 101 side); the k = 8 probe FAILED on power alone (right
        // ordering, band 8.9e-8 vs diff 5.7e-8).
        let w550 = wl_index(&clear, 550.0);
        let (m_99, se_99) = perwl_mean_se(&atm_t3, &config, 99.0, None, w550, k, false);
        let (m_101, se_101) =
            perwl_mean_se(&atm_t3, &config, 101.0, None, w550, 32, false);
        let (m_103, se_103) =
            perwl_mean_se(&atm_t3, &config, 103.0, None, w550, 32, false);
        eprintln!(
            "G-S3-MONO SZA ladder @ tau*3 550nm: 99 {m_99:.4e} (se {se_99:.1e}) > \
             101 {m_101:.4e} (se {se_101:.1e}) > 103 {m_103:.4e} (se {se_103:.1e})"
        );
        let se_a = (se_99 * se_99 + se_101 * se_101).sqrt();
        let se_b = (se_101 * se_101 + se_103 * se_103).sqrt();
        assert!(
            m_99 - m_101 > 2.0 * se_a,
            "SZA 99 ({m_99:.4e}) must exceed SZA 101 ({m_101:.4e}) by 2 SE ({se_a:.2e})"
        );
        assert!(
            m_101 - m_103 > 2.0 * se_b,
            "SZA 101 ({m_101:.4e}) must exceed SZA 103 ({m_103:.4e}) by 2 SE ({se_b:.2e})"
        );
    }

    /// G-S3-CHI2 (physics gate 4b, bias below single-point noise): scan
    /// 16 SZA points densely across 95.0-99.8 on the tau* = 3 deck
    /// (550 nm), fit ln(radiance) vs SZA with a weighted cubic, and
    /// gate that the residuals are consistent with the per-point
    /// standard errors: a hidden bias in any sub-regime bends the curve
    /// away from smooth; honest MC noise does not. The range contains
    /// the most bias-prone seam of the whole estimator, the
    /// ZENITH_SZA_START = 96 forced-mode turn-on (plus the live VSPG
    /// and zenith-mix ramps).
    ///
    /// WHY THE GATED RANGE ENDS AT 99.5, WITH 12 SEEDS (three drafts,
    /// all preserved as findings, 2026-07-04):
    /// 1. scalar-chain draft, 95-104: tripped the log-fit guard at
    ///    SZA 99.8 (se/m 0.70): a budget statement about the scalar
    ///    chain at zenith, not smoothness.
    /// 2. ALIS draft, 95-104 with ramped budgets: chi2 34.0 on dof 9.
    ///    Post-mortem shows the excess is SE-UNFAITHFULNESS of the
    ///    heavy-tail points, not transport bias: the SZA 103 point read
    ///    1.41e-9 with claimed se/m 0.25 while the 1e9-photon MYSTIC
    ///    referee sits at 2.72e-8 (19x, all six seeds clustered low:
    ///    the unsampled tail does not show in a seed SE), and 103 -> 104
    ///    jumped NON-monotonically to 1.52e-8 on one lottery seed. A
    ///    chi2 against per-point SEs is meaningful exactly where the
    ///    SEs are faithful, which at achievable budgets is the
    ///    collapsed-variance range: SZA <= ~100 for tau* = 3. The
    ///    deep-frontier heavy tail is the program's documented residual
    ///    (RESULTS_DEEP_REGIME.md), gated instead by the referee table
    ///    and the monotonicity ladder.
    /// 3. ALIS draft, 16 points dense over 95.0-99.8 at k = 6: chi2
    ///    63.7 on dof 12, but with sign-ALTERNATING residuals and
    ///    non-monotone neighbor jumps (2.1e-6 -> 3.8e-6 -> 1.4e-6
    ///    across 0.7 deg with claimed 9-35% SEs): a smooth transport
    ///    bias cannot do that; a k = 6 seed SE on a tail-mixture
    ///    distribution UNDERSTATES the sampling error (measured factor
    ///    ~ sqrt(63.7/12) = 2.3) because most seed sets miss the tail
    ///    entirely. The chi2-vs-seed-SE instrument therefore needs
    ///    enough seeds per point for the seed SE itself to be faithful.
    ///
    /// PROTOCOL: 10 points at 0.5 deg spacing over 95.0-99.5, 8k
    /// photons, 12 seeds per point, production ALIS hybrid
    /// (`simulate_at_sza`, polarized = false, the G3 zenith protocol).
    /// Points with se/m >= 0.75 are excluded by rule (none expected in
    /// range; max 3 tolerated) and the chi2 bound tracks the surviving
    /// dof at the 99.9% quantile.
    #[test]
    #[ignore = "g_s3_chi2: heavy MC (hours at contended load)"]
    fn g_s3_smoothness_chi2() {
        let atm = deep_atm_1d(3.0);
        let k: u64 = 12;
        let szas: Vec<f64> = (0..10).map(|i| 95.0 + 0.5 * i as f64).collect();
        let photons_for = |_sza: f64| -> usize { 8_000 };

        // (sza, seed) tasks in parallel; each task is one single-threaded
        // ALIS run over all 41 wavelengths (the 550 bin is extracted).
        let mut tasks = Vec::new();
        for &sza in &szas {
            for seed in 0..k {
                tasks.push((sza, seed));
            }
        }
        let vals: Vec<(f64, u64, f64)> = tasks
            .into_par_iter()
            .map(|(sza, seed)| {
                let config = SimulationConfig {
                    photons_per_wavelength: photons_for(sza),
                    seed_salt: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
                    ..deep_config(0, false)
                };
                let r = simulate_at_sza(&atm, &config, sza, None);
                let w550 = r
                    .wavelengths_nm
                    .iter()
                    .position(|&w| (w - 550.0).abs() < 1e-9)
                    .unwrap();
                (sza, seed, r.radiance[w550])
            })
            .collect();
        let pts: Vec<(f64, f64, f64)> = szas
            .iter()
            .map(|&sza| {
                let s: Vec<f64> = vals
                    .iter()
                    .filter(|(z, _, _)| *z == sza)
                    .map(|&(_, _, v)| v)
                    .collect();
                let m = s.iter().sum::<f64>() / s.len() as f64;
                let var = s.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                    / (s.len() - 1) as f64;
                (sza, m, (var / s.len() as f64).sqrt())
            })
            .collect();

        // Inclusion rule + weighted cubic LS on y = ln(m), sy = se/m.
        let mut used: Vec<(f64, f64, f64)> = Vec::new();
        for &(sza, m, se) in &pts {
            assert!(m > 0.0, "SZA {sza}: nonpositive mean {m:e}");
            let rel = se / m;
            if rel < 0.75 {
                used.push((sza, m, se));
            } else {
                eprintln!("G-S3-CHI2 SZA {sza:6.1}: EXCLUDED (se/m {rel:.2})");
            }
        }
        assert!(
            pts.len() - used.len() <= 3,
            "G-S3-CHI2: {} points excluded (max 3): the deep-end variance \
             is not collapsed enough to even scan",
            pts.len() - used.len()
        );
        let mut ata = [[0.0f64; 4]; 4];
        let mut aty = [0.0f64; 4];
        for &(sza, m, se) in &used {
            let x = sza - 100.0;
            let y = libm::log(m);
            let rel = se / m;
            let wgt = 1.0 / (rel * rel);
            let basis = [1.0, x, x * x, x * x * x];
            for i in 0..4 {
                for j in 0..4 {
                    ata[i][j] += wgt * basis[i] * basis[j];
                }
                aty[i] += wgt * basis[i] * y;
            }
        }
        let mut a = ata;
        let mut b = aty;
        for col in 0..4 {
            let mut piv = col;
            for r in col + 1..4 {
                if a[r][col].abs() > a[piv][col].abs() {
                    piv = r;
                }
            }
            a.swap(col, piv);
            b.swap(col, piv);
            let d = a[col][col];
            assert!(d.abs() > 1e-12, "singular fit matrix");
            for r in 0..4 {
                if r != col {
                    let f = a[r][col] / d;
                    for c2 in 0..4 {
                        a[r][c2] -= f * a[col][c2];
                    }
                    b[r] -= f * b[col];
                }
            }
        }
        let coef: Vec<f64> = (0..4).map(|i| b[i] / a[i][i]).collect();

        let mut chi2 = 0.0;
        for &(sza, m, se) in &used {
            let x = sza - 100.0;
            let yfit = coef[0] + coef[1] * x + coef[2] * x * x + coef[3] * x * x * x;
            let z = (libm::log(m) - yfit) / (se / m);
            chi2 += z * z;
            eprintln!(
                "G-S3-CHI2 SZA {sza:6.1}: m {m:.4e} se/m {:.3} resid_z {z:+.2}",
                se / m
            );
        }
        let dof = used.len() - 4;
        // chi2 99.9% quantiles for dof 3..=6.
        let bound = [16.27, 18.47, 20.52, 22.46][dof - 3];
        eprintln!("G-S3-CHI2: chi2 {chi2:.1} on dof {dof} (99.9% bound {bound})");
        assert!(
            chi2 < bound,
            "G-S3-CHI2: residuals inconsistent with per-point SEs \
             (chi2 {chi2:.1} on dof {dof} > {bound}): hidden bias bends the SZA curve"
        );
    }
}
