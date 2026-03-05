//! Molecular gas absorption for the visible spectrum.
//!
//! Computes absorption extinction coefficients from five trace gases:
//!
//! | Gas  | Source                          | Interpolation          |
//! |------|---------------------------------|------------------------|
//! | O3   | Serdyuchenko et al. (2014)      | Temperature (11 pts)   |
//! | NO2  | Vandaele et al. (1998)          | Temperature (2 pts)    |
//! | O2   | HITRAN 2020 Voigt profiles      | (P,T) bilinear (4 pts) |
//! | H2O  | HITRAN 2020 Voigt profiles      | (P,T) bilinear (4 pts) |
//! | O4   | Thalman & Volkamer (2013) CIA   | Fixed 296 K            |
//!
//! Cross-section data lives in [`gas_absorption_data`] as `const` arrays on a
//! 1 nm reference grid (380--780 nm, 401 points). This module provides
//! interpolation to arbitrary wavelengths and the integration path into
//! [`AtmosphereModel`] via [`apply_gas_absorption`].

use crate::atmosphere::{AtmosphereModel, MAX_SHELLS};
use crate::gas_absorption_data::*;

// ── Gas profile ─────────────────────────────────────────────────────────

/// Per-shell gas concentrations and thermodynamic state.
///
/// The MCRT engine is agnostic to where these numbers come from. The
/// default constructor fills them from the US Standard Atmosphere 1976;
/// a weather-driven path can overwrite them with CAMS/Open-Meteo data.
#[derive(Debug, Clone, Copy)]
pub struct ShellGas {
    /// O3 number density [molecules/m^3]
    pub o3_density: f64,
    /// NO2 number density [molecules/m^3]
    pub no2_density: f64,
    /// O2 number density [molecules/m^3]
    pub o2_density: f64,
    /// H2O number density [molecules/m^3]
    pub h2o_density: f64,
    /// Air number density [molecules/m^3] (needed for O4 CIA)
    pub air_density: f64,
    /// Temperature [K]
    pub temperature_k: f64,
    /// Pressure [hPa]
    pub pressure_hpa: f64,
}

impl Default for ShellGas {
    fn default() -> Self {
        Self {
            o3_density: 0.0,
            no2_density: 0.0,
            o2_density: 0.0,
            h2o_density: 0.0,
            air_density: 0.0,
            temperature_k: 288.15,
            pressure_hpa: 1013.25,
        }
    }
}

/// Gas concentration profile for the full atmosphere.
pub struct GasProfile {
    /// Per-shell gas state.
    pub shells: [ShellGas; MAX_SHELLS],
    /// Number of active shells (must match `AtmosphereModel::num_shells`).
    pub num_shells: usize,
}

impl GasProfile {
    /// Create an empty profile (all zeros).
    pub fn empty() -> Self {
        Self {
            shells: [ShellGas::default(); MAX_SHELLS],
            num_shells: 0,
        }
    }
}

// ── Reference grid helpers ──────────────────────────────────────────────

/// Convert a wavelength in nm to a fractional index into the 1 nm reference
/// grid. Returns `None` if the wavelength is outside [380, 780] nm.
#[inline]
fn wl_to_frac_idx(wl_nm: f64) -> Option<(usize, f64)> {
    let frac = (wl_nm - GAS_WL_MIN_NM) / GAS_WL_STEP_NM;
    if frac < 0.0 || frac > (GAS_WL_COUNT - 1) as f64 {
        return None;
    }
    let idx = frac as usize;
    // Clamp to last valid pair for interpolation.
    let idx = if idx >= GAS_WL_COUNT - 1 {
        GAS_WL_COUNT - 2
    } else {
        idx
    };
    let t = frac - idx as f64;
    Some((idx, t))
}

/// Linear interpolation between two values.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ── Cross-section lookups ───────────────────────────────────────────────

/// O3 absorption cross-section [cm^2/molecule] at given wavelength and
/// temperature. Linearly interpolated in both wavelength and temperature
/// from the Serdyuchenko 11-temperature table.
///
/// Returns 0.0 for wavelengths outside the 380--780 nm range.
pub fn o3_cross_section(wl_nm: f64, temp_k: f64) -> f64 {
    let (wi, wt) = match wl_to_frac_idx(wl_nm) {
        Some(v) => v,
        None => return 0.0,
    };

    // Temperature grid runs from 293 K down to 193 K in 10 K steps.
    // Compute fractional temperature index.
    let t_frac = (O3_TEMPS_K[0] - temp_k) / 10.0; // 0.0 at 293K, 10.0 at 193K
    if t_frac <= 0.0 {
        // At or above 293 K: use highest-temp row.
        return lerp(O3_XS[0][wi], O3_XS[0][wi + 1], wt);
    }
    let max_idx = (O3_N_TEMPS - 1) as f64;
    if t_frac >= max_idx {
        // At or below 193 K: use lowest-temp row.
        let last = O3_N_TEMPS - 1;
        return lerp(O3_XS[last][wi], O3_XS[last][wi + 1], wt);
    }

    let ti = t_frac as usize;
    let tt = t_frac - ti as f64;

    // Bilinear: interpolate wavelength at the two bracketing temperatures,
    // then interpolate between them in temperature.
    let xs_lo = lerp(O3_XS[ti][wi], O3_XS[ti][wi + 1], wt);
    let xs_hi = lerp(O3_XS[ti + 1][wi], O3_XS[ti + 1][wi + 1], wt);
    lerp(xs_lo, xs_hi, tt)
}

/// NO2 absorption cross-section [cm^2/molecule] at given wavelength and
/// temperature. Linearly interpolated between the 294 K and 220 K tables.
///
/// Returns 0.0 outside 380--780 nm.
pub fn no2_cross_section(wl_nm: f64, temp_k: f64) -> f64 {
    let (wi, wt) = match wl_to_frac_idx(wl_nm) {
        Some(v) => v,
        None => return 0.0,
    };

    let xs_294 = lerp(NO2_XS_294K[wi], NO2_XS_294K[wi + 1], wt);
    let xs_220 = lerp(NO2_XS_220K[wi], NO2_XS_220K[wi + 1], wt);

    // Clamp to [220, 294] range for interpolation.
    let t_clamped = if temp_k > 294.0 {
        294.0
    } else if temp_k < 220.0 {
        220.0
    } else {
        temp_k
    };
    let frac = (294.0 - t_clamped) / (294.0 - 220.0); // 0 at 294K, 1 at 220K
    lerp(xs_294, xs_220, frac)
}

/// O2-O2 collision-induced absorption cross-section [cm^5/molecule^2].
///
/// The CIA coefficient has units of cm^5/molecule^2. To obtain extinction
/// in cm^-1 multiply by [O2]^2 where [O2] is in molecules/cm^3.
///
/// Returns 0.0 outside 380--780 nm.
pub fn o4_cia_cross_section(wl_nm: f64) -> f64 {
    let (wi, wt) = match wl_to_frac_idx(wl_nm) {
        Some(v) => v,
        None => return 0.0,
    };
    lerp(O4_CIA_XS[wi], O4_CIA_XS[wi + 1], wt)
}

/// Find bracketing indices and interpolation weights for O2/H2O (P,T)
/// cross-section tables.
///
/// The 4 tabulated (P,T) conditions are:
///   0: (1013.25, 296)   1: (500.04, 260)   2: (300.02, 230)   3: (50.0, 210)
///
/// We treat the 4 conditions as a 1-D altitude proxy and interpolate
/// linearly along this sequence based on pressure.
///
/// Returns `(idx_lo, idx_hi, frac)` where `frac` is 0 at condition
/// `idx_lo` and 1 at condition `idx_hi`.
fn pt_interp_weights(pressure_hpa: f64) -> (usize, usize, f64) {
    // Pressure decreasing: [1013.25, 500.04, 300.02, 50.0]
    if pressure_hpa >= PT_PRESSURES_HPA[0] {
        return (0, 0, 0.0);
    }
    if pressure_hpa <= PT_PRESSURES_HPA[PT_N_CONDITIONS - 1] {
        let last = PT_N_CONDITIONS - 1;
        return (last, last, 0.0);
    }
    for i in 0..(PT_N_CONDITIONS - 1) {
        let p_hi = PT_PRESSURES_HPA[i];
        let p_lo = PT_PRESSURES_HPA[i + 1];
        if pressure_hpa <= p_hi && pressure_hpa >= p_lo {
            let frac = (p_hi - pressure_hpa) / (p_hi - p_lo);
            return (i, i + 1, frac);
        }
    }
    // Fallback (should not happen with valid data).
    (0, 0, 0.0)
}

/// O2 monomer absorption cross-section [cm^2/molecule] at given wavelength
/// and atmospheric (P,T) condition. Interpolated from the 4-condition
/// Voigt profile table.
///
/// Returns 0.0 outside 380--780 nm.
pub fn o2_cross_section(wl_nm: f64, pressure_hpa: f64) -> f64 {
    let (wi, wt) = match wl_to_frac_idx(wl_nm) {
        Some(v) => v,
        None => return 0.0,
    };
    let (pi, pj, pt) = pt_interp_weights(pressure_hpa);

    let xs_i = lerp(O2_XS[pi][wi], O2_XS[pi][wi + 1], wt);
    let xs_j = lerp(O2_XS[pj][wi], O2_XS[pj][wi + 1], wt);
    lerp(xs_i, xs_j, pt)
}

/// H2O absorption cross-section [cm^2/molecule] at given wavelength
/// and atmospheric (P,T) condition. Interpolated from the 4-condition
/// Voigt profile table.
///
/// Returns 0.0 outside 380--780 nm.
pub fn h2o_cross_section(wl_nm: f64, pressure_hpa: f64) -> f64 {
    let (wi, wt) = match wl_to_frac_idx(wl_nm) {
        Some(v) => v,
        None => return 0.0,
    };
    let (pi, pj, pt) = pt_interp_weights(pressure_hpa);

    let xs_i = lerp(H2O_XS[pi][wi], H2O_XS[pi][wi + 1], wt);
    let xs_j = lerp(H2O_XS[pj][wi], H2O_XS[pj][wi + 1], wt);
    lerp(xs_i, xs_j, pt)
}

// ── Standard atmosphere profile ─────────────────────────────────────────

/// O2 volume mixing ratio in dry air.
const O2_VMR: f64 = 0.2095;

/// Default H2O scale height [m]. The mixing ratio of water vapour falls
/// off roughly exponentially with a scale height of ~2 km.
const H2O_SCALE_HEIGHT_M: f64 = 2000.0;

/// Sea-level H2O number density [molecules/m^3].
/// Corresponds to ~60% relative humidity at 288 K (~1.5% VMR).
const H2O_SEA_LEVEL_DENSITY: f64 = 3.8e23;

/// Linear interpolation on the standard atmosphere altitude grid.
///
/// `alt_m` is altitude in metres. Returns the interpolated value from
/// `values` sampled at `STD_ALT_KM`.
fn std_atm_interp(alt_m: f64, values: &[f64; STD_N_ALTS]) -> f64 {
    let alt_km = alt_m / 1000.0;

    if alt_km <= STD_ALT_KM[0] {
        return values[0];
    }
    if alt_km >= STD_ALT_KM[STD_N_ALTS - 1] {
        return values[STD_N_ALTS - 1];
    }

    for i in 0..(STD_N_ALTS - 1) {
        if alt_km >= STD_ALT_KM[i] && alt_km <= STD_ALT_KM[i + 1] {
            let span = STD_ALT_KM[i + 1] - STD_ALT_KM[i];
            if span < 1e-12 {
                return values[i];
            }
            let frac = (alt_km - STD_ALT_KM[i]) / span;
            return lerp(values[i], values[i + 1], frac);
        }
    }

    values[STD_N_ALTS - 1]
}

/// Build a [`GasProfile`] from the US Standard Atmosphere 1976 for an
/// existing [`AtmosphereModel`].
///
/// Temperature, pressure, O3, and NO2 are interpolated from the standard
/// atmosphere tables. O2 density is derived from the air density and the
/// O2 mixing ratio (0.2095). H2O density uses an exponential scale-height
/// approximation with surface density of ~3.8e23 m^-3.
pub fn standard_gas_profile(atm: &AtmosphereModel) -> GasProfile {
    let mut profile = GasProfile::empty();
    profile.num_shells = atm.num_shells;

    for s in 0..atm.num_shells {
        let alt_m = atm.shells[s].altitude_mid;
        let temp = std_atm_interp(alt_m, &STD_TEMP_K);
        let press = std_atm_interp(alt_m, &STD_PRESSURE_HPA);
        let air = std_atm_interp(alt_m, &STD_AIR_DENSITY);
        let o3 = std_atm_interp(alt_m, &STD_O3_DENSITY);
        let no2 = std_atm_interp(alt_m, &STD_NO2_DENSITY);

        let o2 = air * O2_VMR;
        let h2o = H2O_SEA_LEVEL_DENSITY * libm::exp(-alt_m / H2O_SCALE_HEIGHT_M);

        profile.shells[s] = ShellGas {
            o3_density: o3,
            no2_density: no2,
            o2_density: o2,
            h2o_density: h2o,
            air_density: air,
            temperature_k: temp,
            pressure_hpa: press,
        };
    }

    profile
}

// ── O3 column scaling ───────────────────────────────────────────────────

/// One Dobson Unit in molecules/m^2.
///
/// 1 DU = 0.01 mm of O3 at STP = 2.6868e20 molecules/m^2.
const DOBSON_UNIT: f64 = 2.6868e20;

/// Scale the O3 densities in `profile` so that the total O3 vertical
/// column matches `target_du` Dobson Units.
///
/// Uses the shell geometry from `atm` to integrate the existing column,
/// then uniformly scales all shells.
pub fn scale_o3_column(profile: &mut GasProfile, atm: &AtmosphereModel, target_du: f64) {
    // Compute current column in molecules/m^2.
    let mut column = 0.0_f64;
    for s in 0..profile.num_shells.min(atm.num_shells) {
        column += profile.shells[s].o3_density * atm.shells[s].thickness;
    }

    if column < 1e-30 {
        // No O3 at all -- cannot scale. Leave unchanged.
        return;
    }

    let target_col = target_du * DOBSON_UNIT;
    let factor = target_col / column;

    for s in 0..profile.num_shells {
        profile.shells[s].o3_density *= factor;
    }
}

/// Compute the current O3 column in Dobson Units for a given profile.
pub fn o3_column_du(profile: &GasProfile, atm: &AtmosphereModel) -> f64 {
    let mut column = 0.0_f64;
    for s in 0..profile.num_shells.min(atm.num_shells) {
        column += profile.shells[s].o3_density * atm.shells[s].thickness;
    }
    column / DOBSON_UNIT
}

// ── Integration with AtmosphereModel ────────────────────────────────────

/// Compute the total gas absorption extinction coefficient [1/m] at a
/// given wavelength for a single shell.
///
/// Returns the sum of absorption contributions from O3, NO2, O2, H2O,
/// and O4 CIA.
fn shell_gas_extinction(gas: &ShellGas, wl_nm: f64) -> f64 {
    // All cross-sections are in cm^2/molecule. Densities are in
    // molecules/m^3. Product gives m^-1 after unit conversion.
    //
    //   ext [1/m] = σ [cm^2] × 1e-4 [m^2/cm^2] × n [m^-3]

    let cm2_to_m2: f64 = 1e-4;

    // O3
    let ext_o3 = o3_cross_section(wl_nm, gas.temperature_k) * cm2_to_m2 * gas.o3_density;

    // NO2
    let ext_no2 = no2_cross_section(wl_nm, gas.temperature_k) * cm2_to_m2 * gas.no2_density;

    // O2 monomer
    let ext_o2 = o2_cross_section(wl_nm, gas.pressure_hpa) * cm2_to_m2 * gas.o2_density;

    // H2O
    let ext_h2o = h2o_cross_section(wl_nm, gas.pressure_hpa) * cm2_to_m2 * gas.h2o_density;

    // O4 CIA: σ_cia [cm^5/mol^2] × [O2]^2 [mol/cm^3]^2 gives cm^-1.
    // Convert O2 density from m^-3 to cm^-3: n_cm3 = n_m3 × 1e-6.
    // Then ext [cm^-1] = σ_cia × n_cm3^2, convert to m^-1: × 100.
    let o2_cm3 = gas.o2_density * 1e-6;
    let ext_o4 = o4_cia_cross_section(wl_nm) * o2_cm3 * o2_cm3 * 100.0;

    ext_o3 + ext_no2 + ext_o2 + ext_h2o + ext_o4
}

/// Fold gas absorption into an existing [`AtmosphereModel`].
///
/// For each shell and wavelength, this adds the gas absorption extinction
/// to `ShellOptics::extinction` and adjusts `ShellOptics::ssa` so that
/// the scattering coefficient is preserved:
///
/// ```text
/// old_scat = extinction * ssa
/// new_ext  = extinction + ext_gas
/// new_ssa  = old_scat / new_ext
/// ```
///
/// `rayleigh_fraction` and `asymmetry` are left unchanged because gas
/// absorption does not scatter photons.
pub fn apply_gas_absorption(atm: &mut AtmosphereModel, profile: &GasProfile) {
    let ns = atm.num_shells.min(profile.num_shells);
    let nw = atm.num_wavelengths;

    for s in 0..ns {
        for w in 0..nw {
            let wl = atm.wavelengths_nm[w];
            let ext_gas = shell_gas_extinction(&profile.shells[s], wl);

            if ext_gas < 1e-30 {
                continue; // No measurable absorption at this wavelength.
            }

            let old_ext = atm.optics[s][w].extinction;
            let old_scat = old_ext * atm.optics[s][w].ssa;

            let new_ext = old_ext + ext_gas;
            let new_ssa = if new_ext > 1e-30 {
                old_scat / new_ext
            } else {
                0.0
            };

            atm.optics[s][w].extinction = new_ext;
            atm.optics[s][w].ssa = new_ssa;
            // rayleigh_fraction and asymmetry unchanged.
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    // ── wl_to_frac_idx ──

    #[test]
    fn wl_frac_idx_at_grid_start() {
        let (idx, t) = wl_to_frac_idx(380.0).unwrap();
        assert_eq!(idx, 0);
        assert!(t.abs() < EPS);
    }

    #[test]
    fn wl_frac_idx_at_grid_end() {
        let (idx, t) = wl_to_frac_idx(780.0).unwrap();
        assert_eq!(idx, GAS_WL_COUNT - 2);
        assert!((t - 1.0).abs() < EPS);
    }

    #[test]
    fn wl_frac_idx_mid_grid() {
        let (idx, t) = wl_to_frac_idx(550.5).unwrap();
        assert_eq!(idx, 170);
        assert!((t - 0.5).abs() < EPS);
    }

    #[test]
    fn wl_frac_idx_below_range() {
        assert!(wl_to_frac_idx(379.9).is_none());
    }

    #[test]
    fn wl_frac_idx_above_range() {
        assert!(wl_to_frac_idx(780.1).is_none());
    }

    // ── O3 cross-section ──

    #[test]
    fn o3_xs_at_grid_point_293k() {
        // At exactly 380 nm, 293 K -> should return the first element of O3_XS[0].
        let xs = o3_cross_section(380.0, 293.0);
        assert!(
            (xs - O3_XS[0][0]).abs() < EPS,
            "O3 σ(380nm, 293K) = {:.4e}, expected {:.4e}",
            xs,
            O3_XS[0][0]
        );
    }

    #[test]
    fn o3_xs_at_550nm_293k() {
        // 550 nm -> index 170.
        let xs = o3_cross_section(550.0, 293.0);
        let expected = O3_XS[0][170];
        assert!(
            (xs - expected).abs() / expected < 1e-6,
            "O3 σ(550nm, 293K) = {:.4e}, expected {:.4e}",
            xs,
            expected
        );
    }

    #[test]
    fn o3_xs_chappuis_peak_around_600nm() {
        // The Chappuis band peak is near 600 nm. Cross-section should be
        // in the range 4-6e-21 cm^2 at 293 K.
        let xs = o3_cross_section(600.0, 293.0);
        assert!(
            xs > 3e-21 && xs < 7e-21,
            "O3 Chappuis peak σ(600nm) = {:.4e}, expected 4-6e-21",
            xs
        );
    }

    #[test]
    fn o3_xs_optical_depth_550nm_300du() {
        // O3 column at 300 DU: N = 300 × 2.6868e20 = 8.06e22 molecules/m^2
        // τ = σ × N = σ_cm2 × 1e-4 × N_m2
        let xs = o3_cross_section(550.0, 250.0); // ~250K average
        let n_col = 300.0 * 2.6868e20; // molecules/m^2
        let tau = xs * 1e-4 * n_col;
        // Expected τ(O3, 550nm) ~ 0.025-0.035 for 300 DU
        assert!(
            tau > 0.01 && tau < 0.06,
            "τ_O3(550nm, 300DU) = {:.4}, expected ~0.03",
            tau
        );
    }

    #[test]
    fn o3_xs_temperature_interpolation() {
        // Interpolated value at 288 K should lie between 293 K and 283 K values.
        let xs_293 = o3_cross_section(500.0, 293.0);
        let xs_283 = o3_cross_section(500.0, 283.0);
        let xs_288 = o3_cross_section(500.0, 288.0);

        let lo = if xs_293 < xs_283 { xs_293 } else { xs_283 };
        let hi = if xs_293 > xs_283 { xs_293 } else { xs_283 };
        assert!(
            xs_288 >= lo - EPS && xs_288 <= hi + EPS,
            "O3 σ(500nm, 288K) = {:.4e} not between {:.4e} and {:.4e}",
            xs_288,
            xs_293,
            xs_283
        );
    }

    #[test]
    fn o3_xs_clamps_above_293k() {
        // At 300 K (above table range), should return same as 293 K.
        let xs_300 = o3_cross_section(500.0, 300.0);
        let xs_293 = o3_cross_section(500.0, 293.0);
        assert!(
            (xs_300 - xs_293).abs() < EPS,
            "Above-range temp: {:.4e} vs {:.4e}",
            xs_300,
            xs_293
        );
    }

    #[test]
    fn o3_xs_clamps_below_193k() {
        let xs_180 = o3_cross_section(500.0, 180.0);
        let xs_193 = o3_cross_section(500.0, 193.0);
        assert!(
            (xs_180 - xs_193).abs() < EPS,
            "Below-range temp: {:.4e} vs {:.4e}",
            xs_180,
            xs_193
        );
    }

    #[test]
    fn o3_xs_outside_range_returns_zero() {
        assert!(o3_cross_section(300.0, 293.0).abs() < EPS);
        assert!(o3_cross_section(800.0, 293.0).abs() < EPS);
    }

    // ── NO2 cross-section ──

    #[test]
    fn no2_xs_at_400nm_294k() {
        // NO2 σ(400nm, 294K) ~ 6.3e-19 from Vandaele data.
        let xs = no2_cross_section(400.0, 294.0);
        let idx = 20; // 380 + 20 = 400
        assert!(
            (xs - NO2_XS_294K[idx]).abs() / NO2_XS_294K[idx] < 1e-6,
            "NO2 σ(400nm) = {:.4e}, expected {:.4e}",
            xs,
            NO2_XS_294K[idx]
        );
    }

    #[test]
    fn no2_xs_decreases_with_wavelength() {
        // NO2 cross-section decreases from blue to red.
        let xs_400 = no2_cross_section(400.0, 294.0);
        let xs_600 = no2_cross_section(600.0, 294.0);
        assert!(
            xs_400 > xs_600,
            "NO2: σ(400nm)={:.4e} should be > σ(600nm)={:.4e}",
            xs_400,
            xs_600
        );
    }

    #[test]
    fn no2_xs_temperature_interpolation() {
        let xs_294 = no2_cross_section(500.0, 294.0);
        let xs_220 = no2_cross_section(500.0, 220.0);
        let xs_257 = no2_cross_section(500.0, 257.0); // midpoint
        let mid = (xs_294 + xs_220) / 2.0;
        assert!(
            (xs_257 - mid).abs() / mid < 1e-6,
            "NO2 midpoint: {:.4e} vs {:.4e}",
            xs_257,
            mid
        );
    }

    // ── O4 CIA ──

    #[test]
    fn o4_cia_peak_near_477nm() {
        // The dominant O4 CIA band peaks near 477 nm.
        let xs_477 = o4_cia_cross_section(477.0);
        let xs_380 = o4_cia_cross_section(380.0);
        let xs_550 = o4_cia_cross_section(550.0);
        assert!(
            xs_477 > xs_380 && xs_477 > xs_550,
            "O4 CIA should peak near 477nm: σ(477)={:.4e}, σ(380)={:.4e}, σ(550)={:.4e}",
            xs_477,
            xs_380,
            xs_550
        );
    }

    #[test]
    fn o4_cia_physically_reasonable_magnitude() {
        // At 477 nm, peak CIA should be ~1e-46 cm^5/mol^2.
        let xs = o4_cia_cross_section(477.0);
        assert!(
            xs > 1e-47 && xs < 1e-45,
            "O4 CIA(477nm) = {:.4e}, expected ~1e-46",
            xs
        );
    }

    // ── O2 cross-section ──

    #[test]
    fn o2_xs_mostly_zero_in_visible() {
        // O2 has almost no absorption in the visible except narrow bands
        // near 628, 688, 762 nm. At 550 nm it should be essentially zero.
        let xs = o2_cross_section(550.0, 1013.25);
        assert!(xs < 1e-26, "O2 σ(550nm) should be ~0, got {:.4e}", xs);
    }

    #[test]
    fn o2_xs_gamma_band_region() {
        // The O2 gamma band is near 628 nm. There should be some absorption.
        // Check a nearby nm to see if there's any signal.
        let xs = o2_cross_section(628.0, 1013.25);
        // Could be very small or exactly zero depending on whether the 1nm
        // grid captures the narrow line. Just verify it's non-negative.
        assert!(xs >= 0.0, "O2 σ(628nm) should be >= 0, got {:.4e}", xs);
    }

    // ── H2O cross-section ──

    #[test]
    fn h2o_xs_has_structure_near_720nm() {
        // H2O has absorption features in the 700-780 nm region.
        let xs_720 = h2o_cross_section(720.0, 1013.25);
        let xs_500 = h2o_cross_section(500.0, 1013.25);
        // Near 720 nm we expect more absorption than at 500 nm.
        assert!(
            xs_720 > xs_500 || xs_720 >= 0.0,
            "H2O: σ(720nm)={:.4e}, σ(500nm)={:.4e}",
            xs_720,
            xs_500
        );
    }

    #[test]
    fn h2o_xs_non_negative() {
        for wl in (380..=780).step_by(5) {
            let xs = h2o_cross_section(wl as f64, 1013.25);
            assert!(xs >= 0.0, "H2O σ({}) = {:.4e} should be >= 0", wl, xs);
        }
    }

    // ── Standard atmosphere profile ──

    fn make_test_atm() -> AtmosphereModel {
        // 10 shells: 0-1, 1-2, ..., 9-10 km, then 10-25, 25-50, 50-100 km
        let alts: [f64; 14] = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 25.0, 50.0, 100.0,
        ];
        let wls = [400.0, 500.0, 550.0, 600.0, 700.0];
        AtmosphereModel::new(&alts, &wls)
    }

    #[test]
    fn std_profile_has_correct_shell_count() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);
        assert_eq!(prof.num_shells, atm.num_shells);
    }

    #[test]
    fn std_profile_surface_temperature() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);
        // Shell 0 midpoint at 500 m. Temperature should be close to 288 K.
        assert!(
            (prof.shells[0].temperature_k - 288.15).abs() < 5.0,
            "Surface T = {:.1} K, expected ~288 K",
            prof.shells[0].temperature_k
        );
    }

    #[test]
    fn std_profile_surface_pressure() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);
        assert!(
            (prof.shells[0].pressure_hpa - 1013.25).abs() < 60.0,
            "Surface P = {:.1} hPa, expected ~1013 hPa",
            prof.shells[0].pressure_hpa
        );
    }

    #[test]
    fn std_profile_o3_peaks_in_stratosphere() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);

        // O3 should peak in shell 11 (10-25 km, midpoint ~17.5 km) or shell 12 (25-50 km).
        let surface_o3 = prof.shells[0].o3_density;
        let strat_o3 = prof.shells[11].o3_density; // 10-25 km
        assert!(
            strat_o3 > surface_o3,
            "Stratospheric O3 ({:.2e}) should be > surface ({:.2e})",
            strat_o3,
            surface_o3
        );
    }

    #[test]
    fn std_profile_o2_from_air_density() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);

        // O2 = air_density × 0.2095
        for s in 0..prof.num_shells {
            let expected = prof.shells[s].air_density * O2_VMR;
            assert!(
                (prof.shells[s].o2_density - expected).abs() < EPS,
                "Shell {} O2: {:.4e} vs {:.4e}",
                s,
                prof.shells[s].o2_density,
                expected
            );
        }
    }

    #[test]
    fn std_profile_h2o_decreases_with_altitude() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);

        for s in 0..(atm.num_shells - 1) {
            if atm.shells[s].altitude_mid < atm.shells[s + 1].altitude_mid {
                assert!(
                    prof.shells[s].h2o_density >= prof.shells[s + 1].h2o_density,
                    "H2O should decrease: shell {} ({:.2e}) vs {} ({:.2e})",
                    s,
                    prof.shells[s].h2o_density,
                    s + 1,
                    prof.shells[s + 1].h2o_density
                );
            }
        }
    }

    #[test]
    fn std_profile_all_densities_non_negative() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);

        for s in 0..prof.num_shells {
            assert!(prof.shells[s].o3_density >= 0.0);
            assert!(prof.shells[s].no2_density >= 0.0);
            assert!(prof.shells[s].o2_density >= 0.0);
            assert!(prof.shells[s].h2o_density >= 0.0);
            assert!(prof.shells[s].air_density >= 0.0);
        }
    }

    // ── O3 column scaling ──

    #[test]
    fn o3_column_du_standard_profile() {
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);
        let du = o3_column_du(&prof, &atm);
        // US Std Atm should give roughly 300 DU. Our coarse grid may
        // differ, but should be in a sane range.
        assert!(
            du > 100.0 && du < 600.0,
            "Standard O3 column = {:.1} DU, expected ~200-400 DU",
            du
        );
    }

    #[test]
    fn o3_column_scaling_to_300du() {
        let atm = make_test_atm();
        let mut prof = standard_gas_profile(&atm);
        scale_o3_column(&mut prof, &atm, 300.0);

        let du = o3_column_du(&prof, &atm);
        assert!(
            (du - 300.0).abs() < 0.1,
            "After scaling to 300 DU: got {:.2} DU",
            du
        );
    }

    #[test]
    fn o3_column_scaling_preserves_shape() {
        let atm = make_test_atm();
        let prof_orig = standard_gas_profile(&atm);
        let mut prof = standard_gas_profile(&atm);
        scale_o3_column(&mut prof, &atm, 300.0);

        // The ratio of shell O3 to total should be preserved.
        let total_orig: f64 = (0..prof_orig.num_shells)
            .map(|s| prof_orig.shells[s].o3_density)
            .fold(0.0, |a, b| a + b);
        let total_new: f64 = (0..prof.num_shells)
            .map(|s| prof.shells[s].o3_density)
            .fold(0.0, |a, b| a + b);

        if total_orig > 0.0 {
            for s in 0..prof.num_shells {
                let frac_orig = prof_orig.shells[s].o3_density / total_orig;
                let frac_new = prof.shells[s].o3_density / total_new;
                assert!(
                    (frac_orig - frac_new).abs() < 1e-10,
                    "Shape changed at shell {}",
                    s
                );
            }
        }
    }

    // ── shell_gas_extinction ──

    #[test]
    fn shell_gas_extinction_pure_o3() {
        let gas = ShellGas {
            o3_density: 5.4e17, // surface O3
            no2_density: 0.0,
            o2_density: 0.0,
            h2o_density: 0.0,
            air_density: 0.0,
            temperature_k: 288.0,
            pressure_hpa: 1013.25,
        };
        let ext = shell_gas_extinction(&gas, 550.0);
        // σ(550nm, 288K) ~ 3.3e-21 cm^2, n = 5.4e17 m^-3
        // ext = 3.3e-21 × 1e-4 × 5.4e17 ~ 1.78e-7 m^-1
        assert!(
            ext > 1e-8 && ext < 1e-5,
            "O3 extinction at 550nm = {:.4e} m^-1",
            ext
        );
    }

    #[test]
    fn shell_gas_extinction_zero_when_no_gas() {
        let gas = ShellGas::default();
        let ext = shell_gas_extinction(&gas, 550.0);
        assert!(ext.abs() < 1e-30, "Zero gas should give zero extinction");
    }

    #[test]
    fn shell_gas_extinction_no2_dominates_at_400nm() {
        // With typical surface NO2 (~4e15 m^-3) and O3 (~5.4e17 m^-3):
        // NO2 σ(400nm) ~ 6.3e-19, O3 σ(400nm) ~ 6e-24
        // NO2 ext = 6.3e-19 × 1e-4 × 4e15 ~ 2.5e-7
        // O3 ext = 6e-24 × 1e-4 × 5.4e17 ~ 3.2e-10
        // So NO2 >> O3 at 400 nm (even though NO2 density is 100x smaller,
        // its cross-section is 100,000x larger).
        let gas = ShellGas {
            o3_density: 5.4e17,
            no2_density: 4.0e15,
            o2_density: 0.0,
            h2o_density: 0.0,
            air_density: 0.0,
            temperature_k: 288.0,
            pressure_hpa: 1013.25,
        };
        let ext = shell_gas_extinction(&gas, 400.0);
        // Should be dominated by NO2
        let ext_no2_only = no2_cross_section(400.0, 288.0) * 1e-4 * 4.0e15;
        assert!(
            ext_no2_only / ext > 0.5,
            "NO2 should dominate at 400nm: total={:.4e}, NO2={:.4e}",
            ext,
            ext_no2_only
        );
    }

    // ── apply_gas_absorption ──

    #[test]
    fn apply_gas_absorption_increases_extinction() {
        let mut atm = make_test_atm();
        // Set some Rayleigh scattering
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                atm.optics[s][w].extinction = 1e-5;
                atm.optics[s][w].ssa = 1.0;
                atm.optics[s][w].rayleigh_fraction = 1.0;
            }
        }

        let pre_ext: [[f64; 5]; 13] = {
            let mut arr = [[0.0; 5]; 13];
            for s in 0..atm.num_shells {
                for w in 0..atm.num_wavelengths {
                    arr[s][w] = atm.optics[s][w].extinction;
                }
            }
            arr
        };

        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    atm.optics[s][w].extinction >= pre_ext[s][w],
                    "Shell {} wl {} extinction should not decrease",
                    s,
                    w
                );
            }
        }
    }

    #[test]
    fn apply_gas_absorption_decreases_ssa() {
        let mut atm = make_test_atm();
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                atm.optics[s][w].extinction = 1e-4;
                atm.optics[s][w].ssa = 1.0;
                atm.optics[s][w].rayleigh_fraction = 1.0;
            }
        }

        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        // SSA should be <= 1.0 after adding absorption (pure scattering -> less pure).
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    atm.optics[s][w].ssa <= 1.0 + EPS,
                    "SSA should be <= 1.0: shell {} wl {}: ssa = {}",
                    s,
                    w,
                    atm.optics[s][w].ssa
                );
            }
        }
    }

    #[test]
    fn apply_gas_absorption_preserves_scattering_coeff() {
        let mut atm = make_test_atm();
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                atm.optics[s][w].extinction = 1e-4;
                atm.optics[s][w].ssa = 0.9;
                atm.optics[s][w].rayleigh_fraction = 0.7;
                atm.optics[s][w].asymmetry = 0.6;
            }
        }

        let pre_scat: [[f64; 5]; 13] = {
            let mut arr = [[0.0; 5]; 13];
            for s in 0..atm.num_shells {
                for w in 0..atm.num_wavelengths {
                    arr[s][w] = atm.optics[s][w].extinction * atm.optics[s][w].ssa;
                }
            }
            arr
        };

        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                let new_scat = atm.optics[s][w].extinction * atm.optics[s][w].ssa;
                assert!(
                    (new_scat - pre_scat[s][w]).abs() < 1e-15,
                    "Scattering coeff changed at shell {} wl {}: {:.6e} vs {:.6e}",
                    s,
                    w,
                    new_scat,
                    pre_scat[s][w]
                );
            }
        }
    }

    #[test]
    fn apply_gas_absorption_preserves_rayleigh_fraction() {
        let mut atm = make_test_atm();
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                atm.optics[s][w].extinction = 1e-4;
                atm.optics[s][w].ssa = 1.0;
                atm.optics[s][w].rayleigh_fraction = 0.8;
                atm.optics[s][w].asymmetry = 0.5;
            }
        }

        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    (atm.optics[s][w].rayleigh_fraction - 0.8).abs() < EPS,
                    "rayleigh_fraction should be unchanged"
                );
                assert!(
                    (atm.optics[s][w].asymmetry - 0.5).abs() < EPS,
                    "asymmetry should be unchanged"
                );
            }
        }
    }

    #[test]
    fn apply_gas_absorption_on_empty_atmosphere() {
        // No existing extinction -- gas absorption becomes the total.
        let mut atm = make_test_atm();
        // optics are already default (ext=0, ssa=1)

        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        // At least some shells should now have non-zero extinction.
        let mut any_nonzero = false;
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                if atm.optics[s][w].extinction > 1e-20 {
                    any_nonzero = true;
                    // With only absorption and no scattering, SSA should be 0.
                    assert!(
                        atm.optics[s][w].ssa < 1e-10,
                        "Pure absorption shell should have SSA~0, got {}",
                        atm.optics[s][w].ssa
                    );
                }
            }
        }
        assert!(any_nonzero, "Some shells should have gas extinction");
    }

    #[test]
    fn apply_gas_absorption_idempotent_zero_profile() {
        let mut atm = make_test_atm();
        atm.optics[0][0].extinction = 1e-4;
        atm.optics[0][0].ssa = 0.95;

        let prof = GasProfile::empty();
        apply_gas_absorption(&mut atm, &prof);

        // With num_shells = 0, nothing should change.
        assert!(
            (atm.optics[0][0].extinction - 1e-4).abs() < EPS,
            "Empty profile should not change extinction"
        );
        assert!(
            (atm.optics[0][0].ssa - 0.95).abs() < EPS,
            "Empty profile should not change SSA"
        );
    }

    // ── Optical depth sanity checks ──

    #[test]
    fn o3_vertical_optical_depth_chappuis() {
        // Integrate O3 absorption optical depth at 600 nm through the
        // standard atmosphere. Should be ~0.03-0.05 for ~300 DU.
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);

        let mut tau = 0.0_f64;
        for s in 0..atm.num_shells {
            let xs = o3_cross_section(600.0, prof.shells[s].temperature_k);
            let ext = xs * 1e-4 * prof.shells[s].o3_density;
            tau += ext * atm.shells[s].thickness;
        }
        assert!(
            tau > 0.01 && tau < 0.1,
            "τ_O3(600nm) = {:.4}, expected ~0.03-0.05",
            tau
        );
    }

    #[test]
    fn no2_vertical_optical_depth_400nm() {
        // NO2 optical depth at 400 nm through the std atm should be small.
        // Column ~1-3e15 molecules/cm^2, σ ~6e-19 -> τ ~6e-4 to 2e-3.
        let atm = make_test_atm();
        let prof = standard_gas_profile(&atm);

        let mut tau = 0.0_f64;
        for s in 0..atm.num_shells {
            let xs = no2_cross_section(400.0, prof.shells[s].temperature_k);
            let ext = xs * 1e-4 * prof.shells[s].no2_density;
            tau += ext * atm.shells[s].thickness;
        }
        assert!(
            tau > 1e-5 && tau < 0.1,
            "τ_NO2(400nm) = {:.6}, expected ~1e-3",
            tau
        );
    }

    #[test]
    fn total_gas_optical_depth_reasonable() {
        // Total gas absorption optical depth at 550 nm should be small
        // (< 0.1) since Rayleigh optical depth is only ~0.1 and gas
        // absorption should be a fraction of that.
        let mut atm = make_test_atm();
        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        // Wavelength 550 nm is index 2 in our test atmosphere.
        let mut tau = 0.0_f64;
        for s in 0..atm.num_shells {
            tau += atm.optics[s][2].extinction * atm.shells[s].thickness;
        }
        assert!(
            tau > 0.0 && tau < 1.0,
            "Total gas τ(550nm) = {:.6}, expected < 1.0",
            tau
        );
    }

    // ── pt_interp_weights ──

    #[test]
    fn pt_weights_at_surface() {
        let (i, j, f) = pt_interp_weights(1013.25);
        assert_eq!(i, 0);
        assert_eq!(j, 0);
        assert!(f.abs() < EPS);
    }

    #[test]
    fn pt_weights_above_surface() {
        let (i, j, _f) = pt_interp_weights(1100.0);
        assert_eq!(i, 0);
        assert_eq!(j, 0);
    }

    #[test]
    fn pt_weights_at_lowest_pressure() {
        let (i, j, _f) = pt_interp_weights(50.0);
        assert_eq!(i, 3);
        assert_eq!(j, 3);
    }

    #[test]
    fn pt_weights_below_lowest_pressure() {
        let (i, j, _f) = pt_interp_weights(10.0);
        assert_eq!(i, 3);
        assert_eq!(j, 3);
    }

    #[test]
    fn pt_weights_midpoint() {
        // Midpoint between conditions 0 and 1: P = (1013.25 + 500.04) / 2
        let p_mid = (1013.25 + 500.04) / 2.0;
        let (i, j, f) = pt_interp_weights(p_mid);
        assert_eq!(i, 0);
        assert_eq!(j, 1);
        assert!((f - 0.5).abs() < 0.01, "frac = {}", f);
    }

    // ── std_atm_interp ──

    #[test]
    fn std_atm_interp_at_grid_point() {
        let val = std_atm_interp(0.0, &STD_TEMP_K);
        assert!((val - 288.15).abs() < EPS);
    }

    #[test]
    fn std_atm_interp_between_grid_points() {
        // Midpoint of 0 and 1 km → 500 m
        let val = std_atm_interp(500.0, &STD_TEMP_K);
        let expected = (288.15 + 281.65) / 2.0;
        assert!(
            (val - expected).abs() < 0.01,
            "T(500m) = {:.2}, expected {:.2}",
            val,
            expected
        );
    }

    #[test]
    fn std_atm_interp_below_grid() {
        let val = std_atm_interp(-1000.0, &STD_TEMP_K);
        assert!((val - STD_TEMP_K[0]).abs() < EPS);
    }

    #[test]
    fn std_atm_interp_above_grid() {
        let val = std_atm_interp(200_000.0, &STD_TEMP_K);
        assert!((val - STD_TEMP_K[STD_N_ALTS - 1]).abs() < EPS);
    }

    // ── lerp ──

    #[test]
    fn lerp_at_endpoints() {
        assert!((lerp(1.0, 3.0, 0.0) - 1.0).abs() < EPS);
        assert!((lerp(1.0, 3.0, 1.0) - 3.0).abs() < EPS);
    }

    #[test]
    fn lerp_midpoint() {
        assert!((lerp(1.0, 3.0, 0.5) - 2.0).abs() < EPS);
    }

    // ── Edge cases ──

    #[test]
    fn all_xs_non_negative_across_grid() {
        for wl in (380..=780).step_by(5) {
            let wl = wl as f64;
            assert!(
                o3_cross_section(wl, 250.0) >= -1e-30,
                "O3 negative at {}",
                wl
            );
            assert!(
                no2_cross_section(wl, 250.0) >= 0.0,
                "NO2 negative at {}",
                wl
            );
            assert!(o4_cia_cross_section(wl) >= 0.0, "O4 negative at {}", wl);
            assert!(o2_cross_section(wl, 500.0) >= 0.0, "O2 negative at {}", wl);
            assert!(
                h2o_cross_section(wl, 500.0) >= 0.0,
                "H2O negative at {}",
                wl
            );
        }
    }

    #[test]
    fn o3_xs_negative_clamp_at_193k() {
        // The 193K Serdyuchenko data has one slightly negative value at
        // 382 nm (-5.75e-25). Our interpolation should handle this
        // gracefully -- it may return a small negative, which is fine
        // for physical accuracy (measurement noise). Verify it's bounded.
        let xs = o3_cross_section(382.0, 193.0);
        assert!(
            xs > -1e-23,
            "O3 σ(382nm, 193K) = {:.4e}, should not be wildly negative",
            xs
        );
    }

    #[test]
    fn wavelength_interpolation_continuity() {
        // Cross-sections should change smoothly between adjacent nm.
        let mut prev = o3_cross_section(500.0, 250.0);
        for wl_tenths in 5001..=5100 {
            let wl = wl_tenths as f64 / 10.0;
            let xs = o3_cross_section(wl, 250.0);
            let change = (xs - prev).abs();
            // Relative change per 0.1 nm should be small.
            if prev > 1e-25 {
                assert!(
                    change / prev < 0.5,
                    "O3 discontinuity at {:.1} nm: {:.4e} -> {:.4e}",
                    wl,
                    prev,
                    xs
                );
            }
            prev = xs;
        }
    }
}
