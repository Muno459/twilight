//! CIE vision functions, spectral analysis, and Fajr/Isha threshold logic.
//!
//! This crate provides:
//! - CIE 1924 photopic V(λ) and CIE 1951 scotopic V'(λ) vision functions
//! - Spectral luminance computation (photopic, scotopic, mesopic per CIE 191:2010)
//! - Spectral centroid analysis for red/white twilight classification
//! - Threshold model for Fajr/Isha prayer time determination

pub mod luminance;
pub mod threshold;
pub mod vision;
