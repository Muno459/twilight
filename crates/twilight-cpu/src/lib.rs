//! Rayon-based CPU backend for parallel photon tracing.

#![allow(clippy::needless_range_loop)] // parallel spectral arrays

#[cfg(feature = "gpu")]
pub mod gpu_dispatch;
pub mod khayt;
pub mod pipeline;
pub mod simulation;
pub mod tracer;
