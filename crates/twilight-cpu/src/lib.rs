#![allow(clippy::needless_range_loop)]
//! Rayon-based CPU backend for parallel photon tracing.

#[cfg(feature = "gpu")]
pub mod gpu_dispatch;
pub mod pipeline;
pub mod simulation;
pub mod tracer;
