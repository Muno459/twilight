// Parallel-array indexing (radiance[w], optics[s][w]) is the dominant
// data layout in this kernel; indexed loops are deliberate and often
// clearer than zipped iterators across 3+ arrays.
#![allow(clippy::needless_range_loop)]

#![no_std]
#![forbid(unsafe_code)]

pub mod atmosphere;
pub mod gas_absorption;
// Generated from tools/gen_gas_xsec.py -- data values that happen to match
// mathematical constants (e.g. 0.3180 ~ 1/pi) are genuine cross-section
// measurements, and the lookup tables must remain `const` for no_std.
#[allow(clippy::approx_constant, clippy::large_const_arrays)]
pub mod gas_absorption_data;
pub mod geometry;
pub mod path_guide;
pub mod photon;
pub mod scattering;
pub mod single_scatter;
pub mod spectrum;
