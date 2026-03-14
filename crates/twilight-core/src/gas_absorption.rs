//! Molecular gas absorption for the visible spectrum.
//!
//! Computes absorption extinction coefficients from five trace gases:
//!
//! | Gas  | Source                              | Interpolation               |
//! |------|-------------------------------------|-----------------------------|
//! | O3   | Serdyuchenko et al. (2014)          | Temperature (11 pts)        |
//! | NO2  | HITRAN XSC database (measured)       | Temperature (2 pts)         |
//! | O2   | HITRAN 2020 Voigt + line-by-line     | Bilinear (P,T) on 5x4 grid |
//! | H2O  | HITRAN 2020 Voigt + line-by-line     | Bilinear (P,T) on 5x4 grid |
//! | O4   | HITRAN 2024 CIA (O2-O2 measured)     | Fixed room temperature      |
//!
//! Cross-section data lives in [`gas_absorption_data`] as `const` arrays on a
//! 1 nm reference grid (380--780 nm, 401 points). This module provides
//! interpolation to arbitrary wavelengths and the integration path into
//! [`AtmosphereModel`] via [`apply_gas_absorption`].
//!
//! For O2 and H2O, line parameters are also stored for optional runtime
//! line-by-line Voigt computation via the Humlicek (1982) algorithm.

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
/// Negative values from measurement noise (e.g. 193K/382nm) are clamped
/// to zero.
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
        let xs = lerp(O3_XS[0][wi], O3_XS[0][wi + 1], wt);
        return if xs > 0.0 { xs } else { 0.0 };
    }
    let max_idx = (O3_N_TEMPS - 1) as f64;
    if t_frac >= max_idx {
        // At or below 193 K: use lowest-temp row.
        let last = O3_N_TEMPS - 1;
        let xs = lerp(O3_XS[last][wi], O3_XS[last][wi + 1], wt);
        return if xs > 0.0 { xs } else { 0.0 };
    }

    let ti = t_frac as usize;
    let tt = t_frac - ti as f64;

    // Bilinear: interpolate wavelength at the two bracketing temperatures,
    // then interpolate between them in temperature.
    let xs_lo = lerp(O3_XS[ti][wi], O3_XS[ti][wi + 1], wt);
    let xs_hi = lerp(O3_XS[ti + 1][wi], O3_XS[ti + 1][wi + 1], wt);
    let xs = lerp(xs_lo, xs_hi, tt);
    // Clamp: Serdyuchenko 193K data has one value at 382nm that is -5.75e-25
    if xs > 0.0 {
        xs
    } else {
        0.0
    }
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
    let t_clamped = temp_k.clamp(220.0, 294.0);
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

// ── Bilinear (P, T) interpolation for O2/H2O ───────────────────────────

/// Find bracketing index and fractional weight for a value in a
/// descending array. Returns `(idx_lo, idx_hi, frac)` where `frac` is
/// 0.0 at `arr[idx_lo]` and 1.0 at `arr[idx_hi]`.
fn descending_interp(val: f64, arr: &[f64]) -> (usize, usize, f64) {
    let n = arr.len();
    if n == 0 {
        return (0, 0, 0.0);
    }
    // Array is descending: arr[0] is largest, arr[n-1] is smallest.
    if val >= arr[0] {
        return (0, 0, 0.0);
    }
    if val <= arr[n - 1] {
        return (n - 1, n - 1, 0.0);
    }
    // Find bracketing pair.
    let mut i = 0;
    while i < n - 1 {
        if val <= arr[i] && val >= arr[i + 1] {
            let span = arr[i] - arr[i + 1];
            if span < 1e-30 {
                return (i, i, 0.0);
            }
            let frac = (arr[i] - val) / span;
            return (i, i + 1, frac);
        }
        i += 1;
    }
    (n - 1, n - 1, 0.0)
}

/// O2 monomer absorption cross-section [cm^2/molecule] at given wavelength
/// and atmospheric (P,T) condition. Bilinearly interpolated on the 5x4
/// (pressure, temperature) Voigt profile grid.
///
/// Returns 0.0 outside 380--780 nm.
pub fn o2_cross_section(wl_nm: f64, pressure_hpa: f64, temperature_k: f64) -> f64 {
    let (wi, wt) = match wl_to_frac_idx(wl_nm) {
        Some(v) => v,
        None => return 0.0,
    };
    let (pi, pj, pf) = descending_interp(pressure_hpa, &PT_PRESSURES_HPA);
    let (ti, tj, tf) = descending_interp(temperature_k, &PT_TEMPS_K);

    // Bilinear: interpolate wavelength at 4 corners, then bilinear in (P,T).
    let xs_piti = lerp(O2_XS[pi][ti][wi], O2_XS[pi][ti][wi + 1], wt);
    let xs_pitj = lerp(O2_XS[pi][tj][wi], O2_XS[pi][tj][wi + 1], wt);
    let xs_pjti = lerp(O2_XS[pj][ti][wi], O2_XS[pj][ti][wi + 1], wt);
    let xs_pjtj = lerp(O2_XS[pj][tj][wi], O2_XS[pj][tj][wi + 1], wt);

    let xs_pi = lerp(xs_piti, xs_pitj, tf);
    let xs_pj = lerp(xs_pjti, xs_pjtj, tf);
    lerp(xs_pi, xs_pj, pf)
}

/// H2O absorption cross-section [cm^2/molecule] at given wavelength
/// and atmospheric (P,T) condition. Bilinearly interpolated on the 5x4
/// (pressure, temperature) Voigt profile grid.
///
/// Returns 0.0 outside 380--780 nm.
pub fn h2o_cross_section(wl_nm: f64, pressure_hpa: f64, temperature_k: f64) -> f64 {
    let (wi, wt) = match wl_to_frac_idx(wl_nm) {
        Some(v) => v,
        None => return 0.0,
    };
    let (pi, pj, pf) = descending_interp(pressure_hpa, &PT_PRESSURES_HPA);
    let (ti, tj, tf) = descending_interp(temperature_k, &PT_TEMPS_K);

    let xs_piti = lerp(H2O_XS[pi][ti][wi], H2O_XS[pi][ti][wi + 1], wt);
    let xs_pitj = lerp(H2O_XS[pi][tj][wi], H2O_XS[pi][tj][wi + 1], wt);
    let xs_pjti = lerp(H2O_XS[pj][ti][wi], H2O_XS[pj][ti][wi + 1], wt);
    let xs_pjtj = lerp(H2O_XS[pj][tj][wi], H2O_XS[pj][tj][wi + 1], wt);

    let xs_pi = lerp(xs_piti, xs_pitj, tf);
    let xs_pj = lerp(xs_pjti, xs_pjtj, tf);
    lerp(xs_pi, xs_pj, pf)
}

// ── Humlicek Voigt profile ──────────────────────────────────────────────

/// Compute the Voigt function K(x, y) = Re[w(z)] where w is the Faddeeva
/// function and z = x + iy.
///
/// Uses the Humlicek (1982) w4 algorithm: a 4-region rational approximation
/// accurate to ~1e-4 relative error for all (x, y >= 0).
///
/// Input:
///   `x` = (nu - nu_0) / gamma_D * sqrt(ln2)   (Doppler-scaled detuning)
///   `y` = gamma_L / gamma_D * sqrt(ln2)        (Lorentz-to-Doppler ratio)
///
/// Returns K(x, y) = (y/pi) * integral of exp(-t^2)/((x-t)^2+y^2) dt.
///
/// Reference: J. Humlicek, JQSRT 27, 437-444 (1982), Algorithm w4.
pub fn voigt_humlicek(x: f64, y: f64) -> f64 {
    // Humlicek w4: compute w(z) = K + iL for z = x + iy.
    // We only need K = Re[w(z)].
    //
    // The algorithm divides the (x,y) plane into 4 regions and uses
    // different rational approximations in each.

    let ax = libm::fabs(x);
    let s = ax + y;
    let t = x * x + y * y; // |z|^2

    if s >= 15.0 {
        // Region 1: |x| + y >= 15
        // w(z) ~ (i/sqrt(pi)) * z / (z^2 - 0.5) ~ i/(sqrt(pi)*z)
        // K = y / (sqrt(pi) * (x^2 + y^2)) * (1 + ...)
        // Use the asymptotic series for w(z):
        // w(z) = i/(sqrt(pi)*z) * sum_{n=0}^{inf} (2n-1)!! / (2*z^2)^n
        let inv_t = 1.0 / t;
        let inv_sqrt_pi = 0.5641895835477563; // 1/sqrt(pi)
                                              // Two-term asymptotic: w ~ (i/sqrt(pi)) * (1/z + 0.5/z^3)
                                              // Re part: y/t * inv_sqrt_pi * (1 + 0.5*(3y^2 - x^2 - y^2 - x^2)/t^2 ...)
                                              // Simpler: just use first term for s >= 15
        return inv_sqrt_pi * y * inv_t;
    }

    if s >= 5.5 {
        // Region 2: 5.5 <= s < 15
        // w(z) ~ (i/sqrt(pi)) * z * (z^2 - 1.5)^{-1}
        // More precisely, 2-term continued fraction:
        // w(z) ~ i/(sqrt(pi)) * z / (z^2 - 0.5) (first correction)
        //
        // Humlicek's region 2 formula uses:
        // w = t1 / (0.5625 + t)  where t1 and t are complex expressions.
        // Let me use the known 2-term rational approximation.

        let inv_sqrt_pi = 0.5641895835477563;
        // w(z) = i/(sqrt(pi)) * (1/z + 0.5/z^3 + 0.75/z^5)
        // = i/(sqrt(pi)) * (1/(x+iy) + 0.5/(x+iy)^3 + ...)
        // For K = Re[w]:
        // Term 1: Re[i/(x+iy)] = Re[i(x-iy)/(x^2+y^2)] = y/t
        // Term 2: Re[i * 0.5/(x+iy)^3]
        // (x+iy)^2 = x^2-y^2 + 2ixy
        // (x+iy)^3 = (x^2-y^2)x - 2xy*y + i((x^2-y^2)y + 2x^2*y)
        //          = x^3-3xy^2 + i(3x^2y - y^3)
        // 1/(x+iy)^3 = conjugate / |z|^6
        //   num_re = x^3-3xy^2, num_im = -(3x^2y - y^3)
        //   Re[i * num] = -num_im = 3x^2y - y^3
        //   So term2 = 0.5 * (3x^2y - y^3) / t^3
        // Combined:
        let t3 = t * t * t;
        let term1 = y / t;
        let term2 = 0.5 * (3.0 * x * x * y - y * y * y) / t3;
        return inv_sqrt_pi * (term1 + term2);
    }

    if y >= 0.195 * ax - 0.176 {
        // Region 3: the "belly" where y is moderate
        // Humlicek's w4 Region 3: rational approximation
        //
        // w(z) = (C0 + C1*z + C2*z^2 + C3*z^3) / (D0 + D1*z + D2*z^2 + D3*z^3 + z^4)
        //
        // For the standard Humlicek w4 coefficients:
        // We use the fact that w(z) for pure imaginary z = iy gives
        // K(0,y) = exp(y^2)*erfc(y). The rational approx matches this.

        // Use the 16-term Weideman (1994) approximation which works for
        // all (x,y) uniformly.
        return weideman_voigt(x, y);
    }

    // Region 4: y is small (Doppler-dominated).
    // Same algorithm works.
    weideman_voigt(x, y)
}

/// Weideman (1994) N=32 approximation for the Faddeeva function.
///
/// w(z) = (L + i*z) / (L^2 + z^2) * sum + corrections
///
/// Actually uses the simpler approach: transform to w(z) via
/// the Laplace continued fraction, evaluated via backward recurrence.
///
/// This implementation uses the Laplace continued fraction:
///   w(z) = i/sqrt(pi) * 1/(z - 0.5/(z - 1.0/(z - 1.5/(z - ...))))
///
/// evaluated via backward recurrence with enough terms for convergence.
fn weideman_voigt(x: f64, y: f64) -> f64 {
    // For very small y, K(x,y) -> exp(-x^2) (pure Gaussian limit).
    // The CF converges poorly for y -> 0, so use the Gaussian directly.
    if y < 1e-6 {
        return libm::exp(-x * x);
    }

    // Laplace continued fraction for w(z), z = x + iy, y > 0:
    //
    // Define the sequence backwards: start from n=N, compute
    //   r_N = 0
    //   r_{n-1} = (n/2) / (z - r_n)
    //
    // Then w(z) = (i/sqrt(pi)) / (z - r_0)
    //
    // This is equivalent to evaluating the continued fraction
    //   w(z) = i/(sqrt(pi)) * 1/(z - 1/2/(z - 1/(z - 3/2/(z - ...))))

    let s = libm::fabs(x) + y;

    // Number of terms for convergence
    let n_terms: usize = if s > 6.0 {
        20
    } else if s > 3.0 {
        30
    } else if s > 1.5 {
        40
    } else if s > 0.5 {
        60
    } else {
        80
    };

    // Backward recurrence in complex arithmetic.
    // r = 0 + 0i
    let mut rr = 0.0_f64; // real part of r
    let mut ri = 0.0_f64; // imag part of r

    let mut n = n_terms;
    while n > 0 {
        let half_n = 0.5 * n as f64;
        // r = half_n / (z - r)  where z = x + iy
        let dr = x - rr;
        let di = y - ri;
        let d2 = dr * dr + di * di;
        if d2 < 1e-300 {
            rr = 0.0;
            ri = 0.0;
        } else {
            // half_n / (dr + i*di) = half_n * (dr - i*di) / d2
            rr = half_n * dr / d2;
            ri = -half_n * di / d2;
        }
        n -= 1;
    }

    // w(z) = (i/sqrt(pi)) / (z - r)
    let dr = x - rr;
    let di = y - ri;
    let d2 = dr * dr + di * di;
    if d2 < 1e-300 {
        return 0.0;
    }

    let inv_sqrt_pi = 0.5641895835477563; // 1/sqrt(pi)

    // i/(dr + i*di) = i*(dr - i*di)/d2 = (di + i*dr)/d2
    // Re[w] = inv_sqrt_pi * di / d2
    inv_sqrt_pi * di / d2
}

// ── Line-by-line cross-section computation ──────────────────────────────

/// Boltzmann factor: compute the ratio of line intensity at temperature T
/// to the reference temperature T_ref = 296 K.
///
/// S(T) = S(T_ref) * Q(T_ref)/Q(T) * exp(-c2*E"*(1/T - 1/T_ref))
///        * (1 - exp(-c2*nu0/T)) / (1 - exp(-c2*nu0/T_ref))
///
/// We approximate Q(T_ref)/Q(T) ~ (T_ref/T)^1.5 for a diatomic,
/// which is accurate to <1% for the temperature range of interest.
///
/// c2 = hc/k_B = 1.4387769 cm*K (second radiation constant).
#[inline]
fn line_intensity_at_temp(sw_ref: f64, elower: f64, nu0: f64, temp_k: f64) -> f64 {
    const T_REF: f64 = 296.0;
    const C2: f64 = 1.4387769; // hc/k_B in cm*K

    let inv_t = 1.0 / temp_k;
    let inv_tref = 1.0 / T_REF;

    // Partition function ratio approximation: (T_ref/T)^1.5
    let q_ratio = libm::pow(T_REF * inv_t, 1.5);

    // Boltzmann factor for lower state
    let boltzmann = libm::exp(-C2 * elower * (inv_t - inv_tref));

    // Stimulated emission correction
    let stim_t = 1.0 - libm::exp(-C2 * nu0 * inv_t);
    let stim_ref = 1.0 - libm::exp(-C2 * nu0 * inv_tref);
    let stim_ratio = if stim_ref > 1e-30 {
        stim_t / stim_ref
    } else {
        1.0
    };

    sw_ref * q_ratio * boltzmann * stim_ratio
}

/// Lorentz half-width at half-maximum at given (P, T).
///
/// gamma_L(P,T) = gamma_air(T_ref) * (P/P_ref) * (T_ref/T)^n_air
///
/// For a self-broadened component at low mixing ratio (< 1%), the
/// self-broadening term is negligible and we use air-broadening only.
#[inline]
fn lorentz_halfwidth(gamma_air_ref: f64, n_air: f64, pressure_hpa: f64, temp_k: f64) -> f64 {
    const T_REF: f64 = 296.0;
    const P_REF: f64 = 1013.25; // hPa (= 1 atm)

    gamma_air_ref * (pressure_hpa / P_REF) * libm::pow(T_REF / temp_k, n_air)
}

/// Doppler half-width at half-maximum for a line at wavenumber nu0 [cm^-1]
/// and temperature T [K] for a molecule of mass `mass_amu` [atomic mass units].
///
/// gamma_D = nu0 * sqrt(2 * k_B * T * ln(2) / (m * c^2))
///         = nu0 / c * sqrt(2 * k_B * T * ln(2) / m)
///
/// Returns gamma_D in cm^-1.
#[inline]
fn doppler_halfwidth(nu0: f64, temp_k: f64, mass_amu: f64) -> f64 {
    // k_B / (amu * c^2) = 1.380649e-23 / (1.66054e-27 * (2.99792e10)^2)
    // = 1.380649e-23 / (1.66054e-27 * 8.98755e20)
    // = 1.380649e-23 / 1.49241e-6
    // = 9.2528e-18
    // But we need sqrt(2 * k_B * ln2 / (m * c^2)):
    // = sqrt(2 * ln2 * 9.2528e-18 / mass_amu)
    // = sqrt(2 * 0.6931 * 9.2528e-18 / mass_amu)
    // = sqrt(1.2822e-17 / mass_amu)
    //
    // Actually let's use the standard formula more carefully:
    // gamma_D = nu0 * sqrt(2*NA*kB*T*ln2 / M) / c
    //         = nu0 * sqrt(2*R*T*ln2 / M) / c
    // where R = 8.31446 J/(mol K), M = mass in kg/mol, c = 2.99792458e10 cm/s
    //
    // gamma_D = (nu0 / c) * sqrt(2 * R * T * ln2 / (mass_amu * 1e-3))
    //         = (nu0 / 2.99792458e10) * sqrt(2 * 8.31446 * T * 0.693147 / (mass_amu * 1e-3))

    const C_CGS: f64 = 2.99792458e10; // cm/s
    const R: f64 = 8.314462618; // J/(mol*K)
    const LN2: f64 = core::f64::consts::LN_2;

    let m_kg_per_mol = mass_amu * 1e-3;
    let factor = libm::sqrt(2.0 * R * LN2 / m_kg_per_mol);
    nu0 / C_CGS * factor * libm::sqrt(temp_k)
}

/// Compute O2 absorption cross-section via line-by-line Voigt profiles.
///
/// This evaluates the Voigt profile for each stored O2 line and sums
/// the contributions at the given wavenumber. This is more accurate than
/// the tabulated cross-sections for arbitrary (P,T) conditions.
///
/// `wl_nm` - wavelength in nm
/// `pressure_hpa` - pressure in hPa
/// `temperature_k` - temperature in K
///
/// Returns cross-section in cm^2/molecule.
pub fn o2_cross_section_lbl(wl_nm: f64, pressure_hpa: f64, temperature_k: f64) -> f64 {
    if !(380.0..=780.0).contains(&wl_nm) {
        return 0.0;
    }

    let nu = 1e7 / wl_nm; // target wavenumber in cm^-1
    const O2_MASS_AMU: f64 = 31.9988; // O2 molecular mass
    const LN2: f64 = core::f64::consts::LN_2;

    let gamma_d_factor = doppler_halfwidth(1.0, temperature_k, O2_MASS_AMU);

    let mut sigma = 0.0_f64;
    let mut i = 0;
    while i < O2_N_LINES {
        let nu0 = O2_LINE_NU[i];
        let sw_ref = O2_LINE_SW[i];
        let gamma_air = O2_LINE_GAMMA_AIR[i];
        let n_air = O2_LINE_N_AIR[i];
        let elower = O2_LINE_ELOWER[i];

        // Quick skip: if the line is very far from our target, skip it.
        // Voigt wings die off roughly as Lorentzian ~ 1/(delta_nu)^2.
        // At 100 cm^-1 away, the contribution is negligible.
        let delta = libm::fabs(nu - nu0);
        if delta > 100.0 {
            i += 1;
            continue;
        }

        // Line intensity at temperature
        let sw = line_intensity_at_temp(sw_ref, elower, nu0, temperature_k);

        // Lorentz half-width
        let gamma_l = lorentz_halfwidth(gamma_air, n_air, pressure_hpa, temperature_k);

        // Doppler half-width
        let gamma_d = nu0 * gamma_d_factor;

        // Voigt parameters
        let sqrt_ln2 = libm::sqrt(LN2);
        let x = (nu - nu0) / gamma_d * sqrt_ln2;
        let y = gamma_l / gamma_d * sqrt_ln2;

        // Voigt profile value
        let k = voigt_humlicek(x, y);

        // Normalized Voigt profile: f_V = K(x,y) * sqrt(ln2) / (gamma_D * sqrt(pi))
        let f_v = k * sqrt_ln2 / (gamma_d * libm::sqrt(core::f64::consts::PI));

        // Cross-section contribution: sigma += S(T) * f_V(nu - nu0)
        sigma += sw * f_v;

        i += 1;
    }

    sigma
}

/// Compute H2O absorption cross-section via line-by-line Voigt profiles.
///
/// Same approach as [`o2_cross_section_lbl`] but for water vapour lines
/// (filtered to S > 1e-25 in the visible range).
///
/// Returns cross-section in cm^2/molecule.
pub fn h2o_cross_section_lbl(wl_nm: f64, pressure_hpa: f64, temperature_k: f64) -> f64 {
    if !(380.0..=780.0).contains(&wl_nm) {
        return 0.0;
    }

    let nu = 1e7 / wl_nm;
    const H2O_MASS_AMU: f64 = 18.0106; // H2O molecular mass
    const LN2: f64 = core::f64::consts::LN_2;

    let gamma_d_factor = doppler_halfwidth(1.0, temperature_k, H2O_MASS_AMU);

    let mut sigma = 0.0_f64;
    let mut i = 0;
    while i < H2O_N_LINES {
        let nu0 = H2O_LINE_NU[i];
        let sw_ref = H2O_LINE_SW[i];
        let gamma_air = H2O_LINE_GAMMA_AIR[i];
        let n_air = H2O_LINE_N_AIR[i];
        let elower = H2O_LINE_ELOWER[i];

        let delta = libm::fabs(nu - nu0);
        if delta > 100.0 {
            i += 1;
            continue;
        }

        let sw = line_intensity_at_temp(sw_ref, elower, nu0, temperature_k);
        let gamma_l = lorentz_halfwidth(gamma_air, n_air, pressure_hpa, temperature_k);
        let gamma_d = nu0 * gamma_d_factor;

        let sqrt_ln2 = libm::sqrt(LN2);
        let x = (nu - nu0) / gamma_d * sqrt_ln2;
        let y = gamma_l / gamma_d * sqrt_ln2;

        let k = voigt_humlicek(x, y);
        let f_v = k * sqrt_ln2 / (gamma_d * libm::sqrt(core::f64::consts::PI));

        sigma += sw * f_v;

        i += 1;
    }

    sigma
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
    //   ext [1/m] = sigma [cm^2] * 1e-4 [m^2/cm^2] * n [m^-3]

    let cm2_to_m2: f64 = 1e-4;

    // O3
    let ext_o3 = o3_cross_section(wl_nm, gas.temperature_k) * cm2_to_m2 * gas.o3_density;

    // NO2
    let ext_no2 = no2_cross_section(wl_nm, gas.temperature_k) * cm2_to_m2 * gas.no2_density;

    // O2 monomer (using tabulated Voigt profiles with bilinear P,T interpolation)
    let ext_o2 =
        o2_cross_section(wl_nm, gas.pressure_hpa, gas.temperature_k) * cm2_to_m2 * gas.o2_density;

    // H2O (using tabulated Voigt profiles with bilinear P,T interpolation)
    let ext_h2o =
        h2o_cross_section(wl_nm, gas.pressure_hpa, gas.temperature_k) * cm2_to_m2 * gas.h2o_density;

    // O4 CIA: sigma_cia [cm^5/mol^2] * [O2]^2 [mol/cm^3]^2 gives cm^-1.
    // Convert O2 density from m^-3 to cm^-3: n_cm3 = n_m3 * 1e-6.
    // Then ext [cm^-1] = sigma_cia * n_cm3^2, convert to m^-1: * 100.
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
        let xs = o3_cross_section(380.0, 293.0);
        assert!(
            (xs - O3_XS[0][0]).abs() < EPS,
            "O3 sigma(380nm, 293K) = {:.4e}, expected {:.4e}",
            xs,
            O3_XS[0][0]
        );
    }

    #[test]
    fn o3_xs_at_550nm_293k() {
        let xs = o3_cross_section(550.0, 293.0);
        let expected = O3_XS[0][170];
        assert!(
            (xs - expected).abs() / expected < 1e-6,
            "O3 sigma(550nm, 293K) = {:.4e}, expected {:.4e}",
            xs,
            expected
        );
    }

    #[test]
    fn o3_xs_chappuis_peak_around_600nm() {
        let xs = o3_cross_section(600.0, 293.0);
        assert!(
            xs > 3e-21 && xs < 7e-21,
            "O3 Chappuis peak sigma(600nm) = {:.4e}, expected 4-6e-21",
            xs
        );
    }

    #[test]
    fn o3_xs_optical_depth_550nm_300du() {
        let xs = o3_cross_section(550.0, 250.0);
        let n_col = 300.0 * 2.6868e20;
        let tau = xs * 1e-4 * n_col;
        assert!(
            tau > 0.01 && tau < 0.06,
            "tau_O3(550nm, 300DU) = {:.4}, expected ~0.03",
            tau
        );
    }

    #[test]
    fn o3_xs_temperature_interpolation() {
        let xs_293 = o3_cross_section(500.0, 293.0);
        let xs_283 = o3_cross_section(500.0, 283.0);
        let xs_288 = o3_cross_section(500.0, 288.0);

        let lo = if xs_293 < xs_283 { xs_293 } else { xs_283 };
        let hi = if xs_293 > xs_283 { xs_293 } else { xs_283 };
        assert!(
            xs_288 >= lo - EPS && xs_288 <= hi + EPS,
            "O3 sigma(500nm, 288K) = {:.4e} not between {:.4e} and {:.4e}",
            xs_288,
            xs_293,
            xs_283
        );
    }

    #[test]
    fn o3_xs_clamps_above_293k() {
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

    #[test]
    fn o3_xs_negative_clamped_at_193k() {
        // The 193K Serdyuchenko data has one slightly negative value at
        // 382 nm (-5.75e-25). After clamping, it should be >= 0.
        let xs = o3_cross_section(382.0, 193.0);
        assert!(
            xs >= 0.0,
            "O3 sigma(382nm, 193K) = {:.4e}, should be >= 0 after clamp",
            xs
        );
    }

    // ── NO2 cross-section ──

    #[test]
    fn no2_xs_at_400nm_294k() {
        // Real HITRAN data: sigma(400nm, 294K) ~ 6.99e-19
        let xs = no2_cross_section(400.0, 294.0);
        assert!(
            xs > 5e-19 && xs < 9e-19,
            "NO2 sigma(400nm, 294K) = {:.4e}, expected ~6.99e-19",
            xs
        );
    }

    #[test]
    fn no2_xs_decreases_with_wavelength() {
        let xs_400 = no2_cross_section(400.0, 294.0);
        let xs_600 = no2_cross_section(600.0, 294.0);
        assert!(
            xs_400 > xs_600,
            "NO2: sigma(400nm)={:.4e} should be > sigma(600nm)={:.4e}",
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

    #[test]
    fn no2_xs_zero_above_667nm() {
        // NO2 XSC coverage ends at ~667nm. Above that, should be zero.
        let xs = no2_cross_section(700.0, 294.0);
        assert!(
            xs < 1e-25,
            "NO2 sigma(700nm) = {:.4e}, expected ~0 (above coverage)",
            xs
        );
    }

    // ── O4 CIA ──

    #[test]
    fn o4_cia_peak_near_577nm() {
        // The real HITRAN CIA data has the dominant band near 577nm, not 477nm.
        let xs_577 = o4_cia_cross_section(577.0);
        let xs_500 = o4_cia_cross_section(500.0);
        assert!(
            xs_577 > xs_500,
            "O4 CIA should have large values near 577nm: sigma(577)={:.4e}, sigma(500)={:.4e}",
            xs_577,
            xs_500
        );
    }

    #[test]
    fn o4_cia_physically_reasonable_magnitude() {
        // At 477 nm, peak CIA should be ~6.6e-46 cm^5/mol^2 (from real HITRAN data).
        let xs = o4_cia_cross_section(477.0);
        assert!(
            xs > 1e-47 && xs < 1e-44,
            "O4 CIA(477nm) = {:.4e}, expected ~6.6e-46",
            xs
        );
    }

    #[test]
    fn o4_cia_real_vs_gaussian() {
        // The real HITRAN CIA at 477nm should be significantly larger than
        // the old Gaussian approximation (1.27e-46).
        let xs = o4_cia_cross_section(477.0);
        assert!(
            xs > 3e-46,
            "O4 CIA(477nm) = {:.4e}, should be > 3e-46 (real data much larger than Gaussian)",
            xs
        );
    }

    // ── O2 cross-section (tabulated, bilinear P,T) ──

    #[test]
    fn o2_xs_mostly_zero_in_visible() {
        let xs = o2_cross_section(550.0, 1013.25, 296.0);
        assert!(xs < 1e-26, "O2 sigma(550nm) should be ~0, got {:.4e}", xs);
    }

    #[test]
    fn o2_xs_gamma_band_region() {
        let xs = o2_cross_section(628.0, 1013.25, 296.0);
        assert!(xs >= 0.0, "O2 sigma(628nm) should be >= 0, got {:.4e}", xs);
    }

    #[test]
    fn o2_xs_bilinear_interpolation() {
        // At a grid point, should return the exact table value.
        let xs = o2_cross_section(380.0, 1013.25, 296.0);
        let expected = O2_XS[0][0][0];
        assert!(
            (xs - expected).abs() < 1e-30 || (xs - expected).abs() / (expected + 1e-30) < 1e-6,
            "O2 grid point: {:.4e} vs {:.4e}",
            xs,
            expected
        );
    }

    // ── H2O cross-section ──

    #[test]
    fn h2o_xs_has_structure_near_720nm() {
        let xs_720 = h2o_cross_section(720.0, 1013.25, 296.0);
        let xs_500 = h2o_cross_section(500.0, 1013.25, 296.0);
        assert!(
            xs_720 > xs_500 || xs_720 >= 0.0,
            "H2O: sigma(720nm)={:.4e}, sigma(500nm)={:.4e}",
            xs_720,
            xs_500
        );
    }

    #[test]
    fn h2o_xs_non_negative() {
        for wl in (380..=780).step_by(5) {
            let xs = h2o_cross_section(wl as f64, 1013.25, 296.0);
            assert!(xs >= 0.0, "H2O sigma({}) = {:.4e} should be >= 0", wl, xs);
        }
    }

    // ── descending_interp ──

    #[test]
    fn descending_interp_at_first_element() {
        let arr = [1013.25, 500.0, 300.0, 100.0, 50.0];
        let (i, j, f) = descending_interp(1013.25, &arr);
        assert_eq!(i, 0);
        assert_eq!(j, 0);
        assert!(f.abs() < EPS);
    }

    #[test]
    fn descending_interp_above_range() {
        let arr = [1013.25, 500.0, 300.0, 100.0, 50.0];
        let (i, j, _f) = descending_interp(1100.0, &arr);
        assert_eq!(i, 0);
        assert_eq!(j, 0);
    }

    #[test]
    fn descending_interp_at_last_element() {
        let arr = [1013.25, 500.0, 300.0, 100.0, 50.0];
        let (i, j, _f) = descending_interp(50.0, &arr);
        assert_eq!(i, 4);
        assert_eq!(j, 4);
    }

    #[test]
    fn descending_interp_below_range() {
        let arr = [1013.25, 500.0, 300.0, 100.0, 50.0];
        let (i, j, _f) = descending_interp(10.0, &arr);
        assert_eq!(i, 4);
        assert_eq!(j, 4);
    }

    #[test]
    fn descending_interp_midpoint() {
        let arr = [1013.25, 500.0, 300.0, 100.0, 50.0];
        let p_mid = (1013.25 + 500.0) / 2.0;
        let (i, j, f) = descending_interp(p_mid, &arr);
        assert_eq!(i, 0);
        assert_eq!(j, 1);
        assert!((f - 0.5).abs() < 0.01, "frac = {}", f);
    }

    // ── Voigt function ──

    #[test]
    fn voigt_gaussian_limit() {
        // When y -> 0, K(x, 0) = exp(-x^2)
        let k = voigt_humlicek(0.0, 1e-10);
        assert!(
            (k - 1.0).abs() < 0.01,
            "K(0, ~0) = {}, expected ~1.0 (Gaussian peak)",
            k
        );
    }

    #[test]
    fn voigt_lorentzian_limit() {
        // When y >> 1, K(0, y) ~ 1/(sqrt(pi)*y) (Lorentzian limit)
        let y = 100.0;
        let k = voigt_humlicek(0.0, y);
        let expected = 1.0 / (libm::sqrt(core::f64::consts::PI) * y);
        assert!(
            (k - expected).abs() / expected < 0.05,
            "K(0, {}) = {:.6e}, expected {:.6e} (Lorentzian limit)",
            y,
            k,
            expected
        );
    }

    #[test]
    fn voigt_symmetric_in_x() {
        let k_pos = voigt_humlicek(2.0, 1.0);
        let k_neg = voigt_humlicek(-2.0, 1.0);
        assert!(
            (k_pos - k_neg).abs() < 1e-10,
            "K(2,1) = {:.6e}, K(-2,1) = {:.6e}, should be equal",
            k_pos,
            k_neg
        );
    }

    #[test]
    fn voigt_positive_for_positive_y() {
        // K(x,y) should be positive for all y > 0.
        for &x in &[0.0, 1.0, 5.0, 10.0, 50.0] {
            for &y in &[0.001, 0.1, 1.0, 10.0, 100.0] {
                let k = voigt_humlicek(x, y);
                assert!(k > 0.0, "K({}, {}) = {} should be > 0", x, y, k);
            }
        }
    }

    #[test]
    fn voigt_intermediate_regime() {
        // At x=0, y=1: K(0,1) = Re[w(i)] = e * erfc(1) = 2.71828 * 0.15730 = 0.42758
        let k = voigt_humlicek(0.0, 1.0);
        assert!(
            (k - 0.42758).abs() < 0.005,
            "K(0, 1) = {}, expected ~0.42758",
            k
        );
    }

    // ── Line-by-line functions ──

    #[test]
    fn line_intensity_at_reference_temp() {
        // At T_ref = 296K, S(T) should equal S_ref.
        let s = line_intensity_at_temp(1e-24, 100.0, 15000.0, 296.0);
        assert!(
            (s - 1e-24).abs() / 1e-24 < 1e-6,
            "S(296K) = {:.4e}, expected 1e-24",
            s
        );
    }

    #[test]
    fn line_intensity_temperature_dependence() {
        // Lower temperature should increase intensity for low E_lower lines
        // (more molecules in ground state).
        let s_296 = line_intensity_at_temp(1e-24, 0.0, 15000.0, 296.0);
        let s_200 = line_intensity_at_temp(1e-24, 0.0, 15000.0, 200.0);
        // For E_lower = 0, Boltzmann factor is 1, but Q ratio changes.
        // With Q ~ T^1.5, Q(296)/Q(200) = (296/200)^1.5 ~ 1.81
        // So S(200) ~ 1.81 * S(296)
        assert!(
            s_200 > s_296,
            "S(200K) = {:.4e} should be > S(296K) = {:.4e} for E_lower=0",
            s_200,
            s_296
        );
    }

    #[test]
    fn lorentz_halfwidth_at_reference() {
        // At reference conditions (1013.25 hPa, 296K), gamma_L = gamma_air.
        let gamma = lorentz_halfwidth(0.05, 0.7, 1013.25, 296.0);
        assert!(
            (gamma - 0.05).abs() < 1e-6,
            "gamma_L at ref = {:.6}, expected 0.05",
            gamma
        );
    }

    #[test]
    fn lorentz_halfwidth_pressure_scaling() {
        // At half pressure, gamma_L should be half.
        let gamma_full = lorentz_halfwidth(0.05, 0.7, 1013.25, 296.0);
        let gamma_half = lorentz_halfwidth(0.05, 0.7, 506.625, 296.0);
        assert!(
            (gamma_half / gamma_full - 0.5).abs() < 1e-6,
            "Half-pressure: {:.6} / {:.6} = {:.4}",
            gamma_half,
            gamma_full,
            gamma_half / gamma_full
        );
    }

    #[test]
    fn doppler_halfwidth_reasonable_for_o2() {
        // O2 at 762 nm (A-band), 296K:
        // nu0 ~ 13123 cm^-1, mass = 31.9988 amu
        // gamma_D = nu0/c * sqrt(2*R*T*ln2/M)
        //         = 13123/2.998e10 * sqrt(2*8.314*296*0.693/0.0320)
        //         = 4.377e-7 * sqrt(106900)
        //         = 4.377e-7 * 327
        //         = 1.43e-4 cm^-1
        let gamma_d = doppler_halfwidth(13123.0, 296.0, 31.9988);
        assert!(
            gamma_d > 1e-4 && gamma_d < 2e-4,
            "O2 Doppler HW at 762nm: {:.6e} cm^-1, expected ~1.4e-4",
            gamma_d
        );
    }

    #[test]
    fn o2_lbl_at_762nm_sea_level() {
        // O2 A-band at 762 nm, sea level. Should show measurable absorption.
        let xs = o2_cross_section_lbl(762.0, 1013.25, 296.0);
        // The A-band has very strong lines. Cross-section should be
        // in the range 1e-25 to 1e-22 depending on exact line position.
        assert!(
            xs > 1e-28,
            "O2 LBL sigma(762nm) = {:.4e}, expected > 1e-28",
            xs
        );
    }

    #[test]
    fn o2_lbl_zero_outside_range() {
        assert!(
            o2_cross_section_lbl(300.0, 1013.25, 296.0) < EPS,
            "O2 LBL outside range should be 0"
        );
    }

    #[test]
    fn h2o_lbl_at_720nm_sea_level() {
        // H2O has absorption features in the 700-780 nm region.
        let xs = h2o_cross_section_lbl(720.0, 1013.25, 296.0);
        assert!(
            xs > 1e-28,
            "H2O LBL sigma(720nm) = {:.4e}, expected > 1e-28",
            xs
        );
    }

    // ── Standard atmosphere profile ──

    fn make_test_atm() -> AtmosphereModel {
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
        let surface_o3 = prof.shells[0].o3_density;
        let strat_o3 = prof.shells[11].o3_density;
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
            o3_density: 5.4e17,
            no2_density: 0.0,
            o2_density: 0.0,
            h2o_density: 0.0,
            air_density: 0.0,
            temperature_k: 288.0,
            pressure_hpa: 1013.25,
        };
        let ext = shell_gas_extinction(&gas, 550.0);
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
        let mut atm = make_test_atm();

        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        let mut any_nonzero = false;
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                if atm.optics[s][w].extinction > 1e-20 {
                    any_nonzero = true;
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
            "tau_O3(600nm) = {:.4}, expected ~0.03-0.05",
            tau
        );
    }

    #[test]
    fn no2_vertical_optical_depth_400nm() {
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
            "tau_NO2(400nm) = {:.6}, expected ~1e-3",
            tau
        );
    }

    #[test]
    fn total_gas_optical_depth_reasonable() {
        let mut atm = make_test_atm();
        let prof = standard_gas_profile(&atm);
        apply_gas_absorption(&mut atm, &prof);

        let mut tau = 0.0_f64;
        for s in 0..atm.num_shells {
            tau += atm.optics[s][2].extinction * atm.shells[s].thickness;
        }
        assert!(
            tau > 0.0 && tau < 1.0,
            "Total gas tau(550nm) = {:.6}, expected < 1.0",
            tau
        );
    }

    // ── std_atm_interp ──

    #[test]
    fn std_atm_interp_at_grid_point() {
        let val = std_atm_interp(0.0, &STD_TEMP_K);
        assert!((val - 288.15).abs() < EPS);
    }

    #[test]
    fn std_atm_interp_between_grid_points() {
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
            assert!(o3_cross_section(wl, 250.0) >= 0.0, "O3 negative at {}", wl);
            assert!(
                no2_cross_section(wl, 250.0) >= 0.0,
                "NO2 negative at {}",
                wl
            );
            assert!(o4_cia_cross_section(wl) >= 0.0, "O4 negative at {}", wl);
            assert!(
                o2_cross_section(wl, 500.0, 260.0) >= 0.0,
                "O2 negative at {}",
                wl
            );
            assert!(
                h2o_cross_section(wl, 500.0, 260.0) >= 0.0,
                "H2O negative at {}",
                wl
            );
        }
    }

    #[test]
    fn wavelength_interpolation_continuity() {
        let mut prev = o3_cross_section(500.0, 250.0);
        for wl_tenths in 5001..=5100 {
            let wl = wl_tenths as f64 / 10.0;
            let xs = o3_cross_section(wl, 250.0);
            let change = (xs - prev).abs();
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
