#![no_std]
#![forbid(unsafe_code)]
#![allow(
    clippy::manual_clamp,
    clippy::manual_memcpy,
    clippy::manual_find,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::doc_overindented_list_items
)]

pub mod atmosphere;
pub mod gas_absorption;
pub mod gas_absorption_data;
pub mod geometry;
pub mod photon;
pub mod scattering;
pub mod single_scatter;
pub mod spectrum;
