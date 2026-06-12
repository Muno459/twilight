//! Satellite cloud data from NASA GIBS (no auth required).
//!
//! Fetches MODIS Aqua Cloud Optical Thickness (COT) and Cloud Top Height
//! (CTH) tiles from the Global Imagery Browse Services WMTS endpoint and
//! decodes the palette PNGs through the official GIBS colormaps (embedded
//! below, generated from colormaps/v1.3/MODIS_VIIRS_*.xml).
//!
//! Geometry: EPSG:4326 "2km" TileMatrixSet, level 5 = 40x20 tiles of
//! 512x512 px, i.e. 9 deg/tile and ~0.0176 deg/px (~2 km).
//!
//! Products are daily (Aqua overpass ~13:30 local); we try today and fall
//! back up to 2 days. Tiles are cached on disk per (layer, date, row, col).
//!
//! The "2.5D" path sampler reads COT at the observer AND at points along
//! the sun azimuth (the twilight light path crosses the cloud field tens
//! to hundreds of km sunward), giving the shadow-path cloud load that a
//! point sample misses.

use std::io::Read;
use std::path::{Path, PathBuf};

const GIBS_BASE: &str = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best";
const COT_LAYER: &str = "MODIS_Aqua_Cloud_Optical_Thickness";
const CTH_LAYER: &str = "MODIS_Aqua_Cloud_Top_Height_Day";
/// Cloud water path [g/m^2] and particle effective radius [um] - the
/// MICROPHYSICS feeds (current through today on GIBS). r_eff types the
/// cloud (droplets ~5-20 um = water; large crystals 20-60 um = ice) and
/// CWP/r_eff yields an independent optical depth (tau = 3 CWP / 2 rho
/// r_eff) that fills pixels where the COT retrieval is missing.
const CWP_LAYER: &str = "MODIS_Aqua_Cloud_Water_Path";
const REFF_LAYER: &str = "MODIS_Aqua_Cloud_Effective_Radius";
const TILE_LEVEL: u32 = 5;
const TILE_DEG: f64 = 9.0;
/// The microphysics layers publish on the finer "1km" TileMatrixSet
/// (level 6: 80x40 tiles, 4.5 deg/tile).
const TILE_LEVEL_1KM: u32 = 6;
const TILE_DEG_1KM: f64 = 4.5;
const TILE_PX: u32 = 512;
const REQUEST_TIMEOUT_MS: u64 = 15_000;

// ── Embedded GIBS colormaps (generated from colormaps/v1.3) ──

#[rustfmt::skip]
pub const COT_COLORMAP: [(u8, u8, u8, f32); 252] = [
    (184, 0, 185, 0.505),
    (182, 0, 184, 1.019),
    (180, 0, 183, 1.058),
    (178, 0, 182, 1.098),
    (176, 0, 181, 1.139),
    (174, 0, 180, 1.182),
    (172, 0, 179, 1.227),
    (170, 0, 178, 1.273),
    (168, 0, 177, 1.321),
    (166, 0, 176, 1.372),
    (164, 0, 175, 1.423),
    (162, 0, 174, 1.478),
    (160, 0, 173, 1.534),
    (158, 0, 172, 1.591),
    (156, 0, 171, 1.651),
    (154, 0, 170, 1.714),
    (152, 0, 169, 1.779),
    (150, 0, 168, 1.846),
    (148, 0, 167, 1.915),
    (146, 0, 166, 1.988),
    (144, 0, 165, 2.063),
    (142, 0, 165, 2.141),
    (140, 0, 164, 2.223),
    (138, 0, 163, 2.306),
    (136, 0, 162, 2.394),
    (134, 0, 161, 2.485),
    (132, 0, 160, 2.579),
    (130, 0, 159, 2.676),
    (128, 0, 158, 2.777),
    (126, 0, 157, 2.883),
    (124, 0, 156, 2.992),
    (122, 0, 155, 3.104),
    (120, 0, 154, 3.222),
    (118, 0, 153, 3.344),
    (116, 0, 152, 3.471),
    (114, 0, 151, 3.602),
    (112, 0, 150, 3.739),
    (110, 0, 149, 3.88),
    (108, 0, 148, 4.027),
    (106, 0, 147, 4.178),
    (104, 0, 146, 4.337),
    (102, 0, 145, 4.5),
    (100, 0, 145, 4.671),
    (97, 0, 147, 4.848),
    (95, 0, 150, 5.031),
    (92, 0, 152, 5.222),
    (90, 3, 155, 5.419),
    (88, 6, 158, 5.624),
    (85, 9, 160, 5.838),
    (83, 12, 163, 6.059),
    (80, 15, 165, 6.287),
    (78, 18, 168, 6.525),
    (76, 22, 171, 6.772),
    (73, 25, 173, 7.028),
    (71, 28, 176, 7.294),
    (69, 31, 179, 7.571),
    (66, 34, 181, 7.857),
    (64, 37, 184, 8.154),
    (61, 40, 186, 8.462),
    (59, 44, 189, 8.783),
    (57, 47, 192, 9.115),
    (54, 50, 194, 9.46),
    (52, 53, 197, 9.817),
    (50, 56, 200, 10.19),
    (47, 59, 202, 10.57),
    (45, 62, 205, 10.98),
    (42, 66, 207, 11.39),
    (40, 69, 210, 11.82),
    (38, 72, 213, 12.27),
    (35, 75, 215, 12.73),
    (33, 78, 218, 13.21),
    (30, 81, 220, 13.71),
    (28, 85, 223, 14.23),
    (26, 88, 226, 14.77),
    (23, 91, 228, 15.33),
    (21, 94, 231, 15.91),
    (19, 97, 234, 16.51),
    (16, 100, 236, 17.14),
    (14, 103, 239, 17.79),
    (11, 107, 241, 18.46),
    (9, 110, 244, 19.16),
    (7, 113, 247, 19.88),
    (4, 116, 249, 20.63),
    (2, 119, 252, 21.42),
    (0, 122, 254, 22.23),
    (0, 125, 248, 23.07),
    (0, 129, 242, 23.94),
    (0, 132, 236, 24.84),
    (0, 135, 230, 25.79),
    (0, 138, 224, 26.76),
    (0, 141, 218, 27.77),
    (0, 144, 212, 28.82),
    (0, 147, 206, 29.91),
    (0, 151, 200, 31.05),
    (0, 154, 194, 32.22),
    (0, 157, 188, 33.44),
    (0, 160, 182, 34.71),
    (0, 163, 176, 36.02),
    (0, 166, 170, 37.38),
    (0, 170, 163, 38.8),
    (0, 173, 157, 40.26),
    (0, 176, 151, 41.79),
    (0, 179, 145, 43.37),
    (0, 182, 139, 45.01),
    (0, 185, 133, 46.71),
    (0, 188, 127, 48.48),
    (0, 192, 121, 50.31),
    (0, 195, 115, 52.22),
    (0, 198, 109, 54.19),
    (0, 201, 103, 56.24),
    (0, 204, 97, 58.37),
    (0, 207, 91, 60.58),
    (0, 210, 85, 62.87),
    (0, 214, 78, 65.25),
    (0, 217, 72, 67.72),
    (0, 220, 66, 70.28),
    (0, 223, 60, 72.94),
    (0, 226, 54, 75.7),
    (0, 229, 48, 78.57),
    (0, 232, 42, 81.54),
    (0, 236, 36, 84.62),
    (0, 239, 30, 87.83),
    (0, 242, 24, 91.15),
    (0, 245, 18, 94.6),
    (0, 248, 12, 98.18),
    (0, 255, 0, 125.0),
    (255, 255, 0, 0.505),
    (255, 255, 4, 1.019),
    (255, 255, 8, 1.058),
    (255, 255, 12, 1.098),
    (255, 255, 17, 1.139),
    (255, 255, 21, 1.182),
    (255, 255, 25, 1.227),
    (255, 255, 30, 1.273),
    (255, 255, 34, 1.321),
    (255, 255, 38, 1.372),
    (255, 255, 42, 1.423),
    (255, 255, 47, 1.478),
    (255, 255, 51, 1.534),
    (255, 255, 55, 1.591),
    (255, 255, 60, 1.651),
    (255, 255, 64, 1.714),
    (255, 255, 68, 1.779),
    (255, 255, 72, 1.846),
    (255, 255, 77, 1.915),
    (255, 255, 81, 1.988),
    (255, 255, 85, 2.063),
    (255, 255, 90, 2.141),
    (255, 255, 94, 2.223),
    (255, 255, 98, 2.306),
    (255, 255, 102, 2.394),
    (255, 255, 107, 2.485),
    (255, 255, 111, 2.579),
    (255, 255, 115, 2.676),
    (255, 255, 120, 2.777),
    (255, 255, 124, 2.883),
    (255, 255, 128, 2.992),
    (255, 255, 132, 3.104),
    (255, 255, 137, 3.222),
    (255, 255, 141, 3.344),
    (255, 255, 145, 3.471),
    (255, 255, 150, 3.602),
    (255, 255, 154, 3.739),
    (255, 255, 158, 3.88),
    (255, 255, 162, 4.027),
    (255, 255, 167, 4.178),
    (255, 255, 171, 4.337),
    (255, 255, 175, 4.5),
    (255, 255, 180, 4.671),
    (255, 255, 181, 4.848),
    (255, 250, 180, 5.031),
    (255, 245, 180, 5.222),
    (255, 240, 180, 5.419),
    (255, 235, 180, 5.624),
    (255, 230, 180, 5.838),
    (255, 230, 181, 6.059),
    (255, 227, 174, 6.287),
    (255, 224, 169, 6.525),
    (255, 221, 164, 6.772),
    (255, 218, 159, 7.028),
    (255, 216, 154, 7.294),
    (255, 213, 149, 7.571),
    (255, 210, 144, 7.857),
    (255, 207, 138, 8.154),
    (255, 205, 133, 8.462),
    (255, 202, 128, 8.783),
    (255, 199, 123, 9.115),
    (255, 196, 118, 9.46),
    (255, 193, 113, 9.817),
    (255, 191, 108, 10.19),
    (255, 188, 102, 10.57),
    (255, 185, 97, 10.98),
    (255, 182, 92, 11.39),
    (255, 180, 87, 11.82),
    (255, 177, 82, 12.27),
    (255, 174, 77, 12.73),
    (255, 171, 72, 13.21),
    (255, 169, 66, 13.71),
    (255, 166, 61, 14.23),
    (255, 163, 56, 14.77),
    (255, 160, 51, 15.33),
    (255, 157, 46, 15.91),
    (255, 155, 41, 16.51),
    (255, 152, 36, 17.14),
    (255, 149, 30, 17.79),
    (255, 146, 25, 18.46),
    (255, 144, 20, 19.16),
    (255, 141, 15, 19.88),
    (255, 138, 10, 20.63),
    (255, 135, 5, 21.42),
    (255, 133, 2, 22.23),
    (255, 133, 0, 23.07),
    (255, 125, 0, 23.94),
    (255, 118, 0, 24.84),
    (255, 110, 0, 25.79),
    (255, 103, 0, 26.76),
    (255, 96, 0, 27.77),
    (255, 88, 0, 28.82),
    (255, 81, 0, 29.91),
    (255, 73, 0, 31.05),
    (255, 66, 0, 32.22),
    (255, 59, 0, 33.44),
    (255, 51, 0, 34.71),
    (255, 44, 0, 36.02),
    (255, 36, 0, 37.38),
    (255, 29, 0, 38.8),
    (255, 22, 0, 40.26),
    (255, 14, 0, 41.79),
    (255, 7, 0, 43.37),
    (255, 3, 0, 45.01),
    (255, 0, 0, 46.71),
    (249, 0, 0, 48.48),
    (243, 0, 0, 50.31),
    (237, 0, 0, 52.22),
    (232, 0, 0, 54.19),
    (226, 0, 0, 56.24),
    (220, 0, 0, 58.37),
    (215, 0, 0, 60.58),
    (209, 0, 0, 62.87),
    (203, 0, 0, 65.25),
    (198, 0, 0, 67.72),
    (192, 0, 0, 70.28),
    (186, 0, 0, 72.94),
    (181, 0, 0, 75.7),
    (175, 0, 0, 78.57),
    (169, 0, 0, 81.54),
    (164, 0, 0, 84.62),
    (158, 0, 0, 87.83),
    (152, 0, 0, 91.15),
    (147, 0, 0, 94.6),
    (141, 0, 0, 98.18),
    (135, 0, 0, 125.0),
];

#[rustfmt::skip]
pub const CTH_COLORMAP_M: [(u8, u8, u8, f32); 240] = [
    (255, 0, 0, 25.0),
    (255, 0, 1, 75.0),
    (255, 1, 0, 125.0),
    (255, 1, 1, 175.0),
    (254, 0, 0, 225.0),
    (254, 0, 1, 275.0),
    (254, 1, 0, 325.0),
    (254, 1, 1, 375.0),
    (254, 2, 1, 425.0),
    (254, 2, 0, 475.0),
    (254, 2, 2, 525.0),
    (253, 0, 0, 575.0),
    (253, 0, 1, 625.0),
    (253, 1, 0, 675.0),
    (253, 1, 1, 725.0),
    (253, 1, 2, 775.0),
    (170, 0, 0, 825.0),
    (170, 0, 1, 875.0),
    (170, 1, 0, 925.0),
    (170, 1, 1, 975.0),
    (171, 0, 0, 1.025e+03),
    (171, 0, 1, 1.075e+03),
    (171, 1, 0, 1.125e+03),
    (171, 1, 1, 1.175e+03),
    (171, 2, 1, 1.225e+03),
    (171, 2, 0, 1.275e+03),
    (171, 2, 2, 1.325e+03),
    (172, 0, 0, 1.375e+03),
    (172, 0, 1, 1.425e+03),
    (172, 1, 0, 1.475e+03),
    (172, 1, 1, 1.525e+03),
    (172, 1, 2, 1.575e+03),
    (110, 0, 0, 1.625e+03),
    (110, 0, 1, 1.675e+03),
    (110, 1, 0, 1.725e+03),
    (110, 1, 1, 1.775e+03),
    (111, 0, 0, 1.825e+03),
    (111, 0, 1, 1.875e+03),
    (111, 1, 0, 1.925e+03),
    (111, 1, 1, 1.975e+03),
    (111, 2, 1, 2.025e+03),
    (111, 2, 0, 2.075e+03),
    (111, 2, 2, 2.125e+03),
    (112, 0, 0, 2.175e+03),
    (112, 0, 1, 2.225e+03),
    (112, 1, 0, 2.275e+03),
    (112, 1, 1, 2.325e+03),
    (112, 1, 2, 2.375e+03),
    (122, 90, 3, 2.425e+03),
    (122, 90, 4, 2.475e+03),
    (122, 91, 3, 2.525e+03),
    (122, 91, 4, 2.575e+03),
    (123, 90, 3, 2.625e+03),
    (123, 90, 4, 2.675e+03),
    (123, 91, 3, 2.725e+03),
    (123, 91, 4, 2.775e+03),
    (123, 92, 4, 2.825e+03),
    (123, 92, 3, 2.875e+03),
    (123, 92, 5, 2.925e+03),
    (124, 90, 3, 2.975e+03),
    (124, 90, 4, 3.025e+03),
    (124, 91, 3, 3.075e+03),
    (124, 91, 4, 3.125e+03),
    (124, 91, 5, 3.175e+03),
    (187, 136, 0, 3.225e+03),
    (187, 136, 1, 3.275e+03),
    (187, 137, 0, 3.325e+03),
    (187, 137, 1, 3.375e+03),
    (188, 136, 0, 3.425e+03),
    (188, 136, 1, 3.475e+03),
    (188, 137, 0, 3.525e+03),
    (188, 137, 1, 3.575e+03),
    (188, 138, 1, 3.625e+03),
    (188, 138, 0, 3.675e+03),
    (188, 138, 2, 3.725e+03),
    (189, 136, 0, 3.775e+03),
    (189, 136, 1, 3.825e+03),
    (189, 137, 0, 3.875e+03),
    (189, 137, 1, 3.925e+03),
    (189, 137, 2, 3.975e+03),
    (240, 190, 64, 4.025e+03),
    (240, 190, 65, 4.075e+03),
    (240, 191, 64, 4.125e+03),
    (240, 191, 65, 4.175e+03),
    (241, 190, 64, 4.225e+03),
    (241, 190, 65, 4.275e+03),
    (241, 191, 64, 4.325e+03),
    (241, 191, 65, 4.375e+03),
    (241, 192, 65, 4.425e+03),
    (241, 192, 64, 4.475e+03),
    (241, 192, 66, 4.525e+03),
    (242, 190, 64, 4.575e+03),
    (242, 190, 65, 4.625e+03),
    (242, 191, 64, 4.675e+03),
    (242, 191, 65, 4.725e+03),
    (242, 191, 66, 4.775e+03),
    (255, 255, 0, 4.825e+03),
    (255, 255, 1, 4.875e+03),
    (255, 254, 0, 4.925e+03),
    (255, 254, 1, 4.975e+03),
    (254, 255, 0, 5.025e+03),
    (254, 255, 1, 5.075e+03),
    (254, 254, 0, 5.125e+03),
    (254, 254, 1, 5.175e+03),
    (254, 253, 1, 5.225e+03),
    (254, 253, 0, 5.275e+03),
    (254, 253, 2, 5.325e+03),
    (253, 255, 0, 5.375e+03),
    (253, 255, 1, 5.425e+03),
    (253, 254, 0, 5.475e+03),
    (253, 254, 1, 5.525e+03),
    (253, 254, 2, 5.575e+03),
    (0, 220, 0, 5.625e+03),
    (0, 220, 1, 5.675e+03),
    (0, 221, 0, 5.725e+03),
    (0, 221, 1, 5.775e+03),
    (1, 220, 0, 5.825e+03),
    (1, 220, 1, 5.875e+03),
    (1, 221, 0, 5.925e+03),
    (1, 221, 1, 5.975e+03),
    (1, 222, 1, 6.025e+03),
    (1, 222, 0, 6.075e+03),
    (1, 222, 2, 6.125e+03),
    (2, 220, 0, 6.175e+03),
    (2, 220, 1, 6.225e+03),
    (2, 221, 0, 6.275e+03),
    (2, 221, 1, 6.325e+03),
    (2, 221, 2, 6.375e+03),
    (0, 136, 0, 6.425e+03),
    (0, 136, 1, 6.475e+03),
    (0, 137, 0, 6.525e+03),
    (0, 137, 1, 6.575e+03),
    (1, 136, 0, 6.625e+03),
    (1, 136, 1, 6.675e+03),
    (1, 137, 0, 6.725e+03),
    (1, 137, 1, 6.775e+03),
    (1, 138, 1, 6.825e+03),
    (1, 138, 0, 6.875e+03),
    (1, 138, 2, 6.925e+03),
    (2, 136, 0, 6.975e+03),
    (2, 136, 1, 7.025e+03),
    (2, 137, 0, 7.075e+03),
    (2, 137, 1, 7.125e+03),
    (2, 137, 2, 7.175e+03),
    (0, 80, 0, 7.225e+03),
    (0, 80, 1, 7.275e+03),
    (0, 81, 0, 7.325e+03),
    (0, 81, 1, 7.375e+03),
    (1, 80, 0, 7.425e+03),
    (1, 80, 1, 7.475e+03),
    (1, 81, 0, 7.525e+03),
    (1, 81, 1, 7.575e+03),
    (1, 82, 1, 7.625e+03),
    (1, 82, 0, 7.675e+03),
    (1, 82, 2, 7.725e+03),
    (2, 80, 0, 7.775e+03),
    (2, 80, 1, 7.825e+03),
    (2, 81, 0, 7.875e+03),
    (2, 81, 1, 7.925e+03),
    (2, 81, 2, 7.975e+03),
    (0, 136, 238, 8.025e+03),
    (0, 136, 239, 8.075e+03),
    (0, 137, 238, 8.125e+03),
    (0, 137, 239, 8.175e+03),
    (1, 136, 238, 8.225e+03),
    (1, 136, 239, 8.275e+03),
    (1, 137, 238, 8.325e+03),
    (1, 137, 239, 8.375e+03),
    (1, 138, 239, 8.425e+03),
    (1, 138, 238, 8.475e+03),
    (1, 138, 240, 8.525e+03),
    (2, 136, 238, 8.575e+03),
    (2, 136, 239, 8.625e+03),
    (2, 137, 238, 8.675e+03),
    (2, 137, 239, 8.725e+03),
    (2, 137, 240, 8.775e+03),
    (0, 0, 255, 8.825e+03),
    (0, 0, 254, 8.875e+03),
    (0, 1, 255, 8.925e+03),
    (0, 1, 254, 8.975e+03),
    (1, 0, 255, 9.025e+03),
    (1, 0, 254, 9.075e+03),
    (1, 1, 255, 9.125e+03),
    (1, 1, 254, 9.175e+03),
    (1, 2, 254, 9.225e+03),
    (1, 2, 255, 9.275e+03),
    (1, 2, 253, 9.325e+03),
    (2, 0, 253, 9.375e+03),
    (2, 0, 254, 9.425e+03),
    (2, 1, 253, 9.475e+03),
    (2, 1, 254, 9.525e+03),
    (2, 1, 255, 9.575e+03),
    (0, 0, 170, 9.625e+03),
    (0, 0, 171, 9.675e+03),
    (0, 1, 170, 9.725e+03),
    (0, 1, 171, 9.775e+03),
    (1, 0, 170, 9.825e+03),
    (1, 0, 171, 9.875e+03),
    (1, 1, 170, 9.925e+03),
    (1, 1, 171, 9.975e+03),
    (1, 2, 171, 1.002e+04),
    (1, 2, 170, 1.008e+04),
    (1, 2, 172, 1.012e+04),
    (2, 0, 170, 1.018e+04),
    (2, 0, 171, 1.022e+04),
    (2, 1, 170, 1.028e+04),
    (2, 1, 171, 1.032e+04),
    (2, 1, 172, 1.038e+04),
    (0, 0, 100, 1.042e+04),
    (0, 0, 101, 1.048e+04),
    (0, 1, 100, 1.052e+04),
    (0, 1, 101, 1.058e+04),
    (1, 0, 100, 1.062e+04),
    (1, 0, 101, 1.068e+04),
    (1, 1, 100, 1.072e+04),
    (1, 1, 101, 1.078e+04),
    (1, 2, 101, 1.082e+04),
    (1, 2, 100, 1.088e+04),
    (1, 2, 102, 1.092e+04),
    (2, 0, 100, 1.098e+04),
    (2, 0, 101, 1.102e+04),
    (2, 1, 100, 1.108e+04),
    (2, 1, 101, 1.112e+04),
    (2, 1, 102, 1.118e+04),
    (183, 15, 141, 1.122e+04),
    (183, 15, 142, 1.128e+04),
    (183, 16, 141, 1.132e+04),
    (183, 16, 142, 1.138e+04),
    (184, 15, 141, 1.142e+04),
    (184, 15, 142, 1.148e+04),
    (184, 16, 141, 1.152e+04),
    (184, 16, 142, 1.158e+04),
    (184, 17, 142, 1.162e+04),
    (184, 17, 141, 1.168e+04),
    (184, 17, 143, 1.172e+04),
    (185, 15, 141, 1.178e+04),
    (185, 15, 142, 1.182e+04),
    (185, 16, 141, 1.188e+04),
    (185, 16, 142, 1.192e+04),
    (185, 16, 143, 1.198e+04),
];


/// Point sample of the satellite cloud field.
#[derive(Debug, Clone, Copy)]
pub struct SatelliteCloud {
    /// Cloud optical thickness (550 nm equivalent) at the sample point.
    pub cot: f64,
    /// Cloud top height [m], when the CTH product has data here.
    pub cloud_top_m: Option<f64>,
    /// Cloud water path [g/m^2] (liquid or ice, per the retrieval phase).
    pub cwp_g_m2: Option<f64>,
    /// Particle effective radius [um] - measured microphysics: ~5-20 um
    /// means liquid droplets, larger means ice crystals.
    pub r_eff_um: Option<f64>,
    /// Age of the product in days (0 = today's overpass).
    pub age_days: u32,
}

impl SatelliteCloud {
    /// Optical depth derived from the microphysics pair:
    /// tau = 3 CWP / (2 rho_w r_eff) = 1.5 CWP[g/m^2] / r_eff[um]
    /// (geometric-optics extinction efficiency ~2; rho_w = 1e6 g/m^3).
    pub fn microphysics_tau(&self) -> Option<f64> {
        match (self.cwp_g_m2, self.r_eff_um) {
            (Some(cwp), Some(re)) if re > 1.0 => Some(1.5 * cwp / re),
            _ => None,
        }
    }
}

/// Observer + sunward-path composite (the "2.5D" sample).
#[derive(Debug, Clone)]
pub struct SatelliteCloudPath {
    /// Sample at the observer (None = clear or no data at observer).
    pub observer: Option<SatelliteCloud>,
    /// Mean COT over the sunward samples that had cloud (0 if none).
    pub path_mean_cot: f64,
    /// Fraction of sunward samples that were cloudy.
    pub path_cloud_fraction: f64,
    /// Number of sunward samples taken.
    pub n_path_samples: usize,
}

fn tile_for(lat: f64, lon: f64) -> (u32, u32, u32, u32) {
    tile_for_grid(lat, lon, TILE_DEG)
}

fn tile_for_grid(lat: f64, lon: f64, tile_deg: f64) -> (u32, u32, u32, u32) {
    let lon = lon.rem_euclid(360.0);
    let lon = if lon > 180.0 { lon - 360.0 } else { lon };
    let max_col = (360.0 / tile_deg) as i64 - 1;
    let max_row = (180.0 / tile_deg) as i64 - 1;
    let col = (((lon + 180.0) / tile_deg).floor() as i64).clamp(0, max_col) as u32;
    let row = (((90.0 - lat) / tile_deg).floor() as i64).clamp(0, max_row) as u32;
    let px = ((((lon + 180.0) % tile_deg) / tile_deg * TILE_PX as f64) as i64)
        .clamp(0, TILE_PX as i64 - 1) as u32;
    let py = ((((90.0 - lat) % tile_deg) / tile_deg * TILE_PX as f64) as i64)
        .clamp(0, TILE_PX as i64 - 1) as u32;
    (row, col, px, py)
}

fn cache_path(cache_dir: &Path, layer: &str, date: &str, row: u32, col: u32) -> PathBuf {
    cache_dir.join(format!("{layer}_{date}_{row}_{col}.png"))
}

fn fetch_tile(
    cache_dir: &Path,
    layer: &str,
    date: &str,
    row: u32,
    col: u32,
) -> Result<Vec<u8>, String> {
    let path = cache_path(cache_dir, layer, date, row, col);
    if let Ok(bytes) = std::fs::read(&path) {
        if !bytes.is_empty() {
            return Ok(bytes);
        }
    }
    // The microphysics layers live on the finer 1km TileMatrixSet.
    let (tms, level) = if layer == CWP_LAYER || layer == REFF_LAYER {
        ("1km", TILE_LEVEL_1KM)
    } else {
        ("2km", TILE_LEVEL)
    };
    let url = format!("{GIBS_BASE}/{layer}/default/{date}/{tms}/{level}/{row}/{col}.png");
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_millis(REQUEST_TIMEOUT_MS)))
        .build()
        .new_agent();
    let mut resp = agent
        .get(&url)
        .call()
        .map_err(|e| format!("GIBS fetch failed: {e}"))?;
    let mut bytes = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("GIBS read failed: {e}"))?;
    let _ = std::fs::create_dir_all(cache_dir);
    let _ = std::fs::write(&path, &bytes);
    Ok(bytes)
}

/// Decode a palette/RGBA PNG and return (rgba_pixels, width, height).
fn decode_png(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| format!("png: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("png: {e}"))?;
    let (w, h) = (info.width, info.height);
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut out = Vec::with_capacity((w * h * 4) as usize);
            for c in buf.chunks_exact(3) {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            out
        }
        other => return Err(format!("unexpected png color type {other:?}")),
    };
    Ok((rgba, w, h))
}

/// Exact-match LUT lookup with a small tolerance for palette rounding.
fn lut_lookup(lut: &[(u8, u8, u8, f32)], r: u8, g: u8, b: u8) -> Option<f64> {
    let mut best: Option<(u32, f32)> = None;
    for &(lr, lg, lb, v) in lut {
        let d = (lr as i32 - r as i32).pow(2) as u32
            + (lg as i32 - g as i32).pow(2) as u32
            + (lb as i32 - b as i32).pow(2) as u32;
        if d == 0 {
            return Some(v as f64);
        }
        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, v));
        }
    }
    // Accept tiny palette rounding only; anything else is not cloud data.
    best.and_then(|(d, v)| if d <= 12 { Some(v as f64) } else { None })
}

/// Outcome of sampling one layer pixel - distinguishes "this pixel is
/// clear" from "this date's tile has no data at all" (GIBS serves a fully
/// transparent tile before the daily product is ingested; conflating the
/// two froze the date-fallback loop on the empty tile).
enum LayerSample {
    Value(f64),
    ClearPixel,
    EmptyTile,
}

fn sample_layer(
    cache_dir: &Path,
    layer: &str,
    lut: &[(u8, u8, u8, f32)],
    date: &str,
    lat: f64,
    lon: f64,
) -> Result<LayerSample, String> {
    let tile_deg = if layer == CWP_LAYER || layer == REFF_LAYER {
        TILE_DEG_1KM
    } else {
        TILE_DEG
    };
    let (row, col, px, py) = tile_for_grid(lat, lon, tile_deg);
    let bytes = fetch_tile(cache_dir, layer, date, row, col)?;
    let (rgba, w, _h) = decode_png(&bytes)?;
    let idx = ((py * w + px) * 4) as usize;
    if idx + 3 >= rgba.len() {
        return Ok(LayerSample::EmptyTile);
    }
    let (r, g, b, a) = (rgba[idx], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]);
    if a == 0 {
        // Transparent at our pixel: clear sky only if the tile carries
        // data SOMEWHERE; a tile with no opaque pixel at all has simply
        // not been ingested for this date yet.
        let any_data = rgba.chunks_exact(4).any(|c| c[3] != 0);
        return Ok(if any_data {
            LayerSample::ClearPixel
        } else {
            LayerSample::EmptyTile
        });
    }
    Ok(match lut_lookup(lut, r, g, b) {
        Some(v) => LayerSample::Value(v),
        None => LayerSample::ClearPixel,
    })
}

/// Convenience: value-or-None for secondary layers (CTH, CWP, r_eff).
fn sample_value(
    cache_dir: &Path,
    layer: &str,
    lut: &[(u8, u8, u8, f32)],
    date: &str,
    lat: f64,
    lon: f64,
) -> Option<f64> {
    match sample_layer(cache_dir, layer, lut, date, lat, lon) {
        Ok(LayerSample::Value(v)) => Some(v),
        _ => None,
    }
}

fn date_minus(date: &str, days: u32) -> Option<String> {
    let mut p = date.split('-');
    let (y, m, d) = (
        p.next()?.parse::<i32>().ok()?,
        p.next()?.parse::<u32>().ok()?,
        p.next()?.parse::<u32>().ok()?,
    );
    // simple civil-date decrement
    let mut y = y;
    let mut m = m;
    let mut d = d as i64 - days as i64;
    let leap = |y: i32| (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let mlen = |y: i32, m: u32| -> i64 {
        match m {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => if leap(y) { 29 } else { 28 },
            _ => 30,
        }
    };
    while d < 1 {
        if m == 1 {
            m = 12;
            y -= 1;
        } else {
            m -= 1;
        }
        d += mlen(y, m);
    }
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

/// Sample the satellite cloud field at a point, trying today then up to
/// 2 days back (MODIS daily products lag the overpass).
pub fn sample_cloud(
    cache_dir: &Path,
    date: &str,
    lat: f64,
    lon: f64,
) -> Option<SatelliteCloud> {
    for age in 0..3u32 {
        let d = if age == 0 {
            date.to_string()
        } else {
            date_minus(date, age)?
        };
        let attach = |cot: f64| -> SatelliteCloud {
            use crate::satellite_colormaps::{CWP_COLORMAP, REFF_COLORMAP};
            let cth = sample_value(cache_dir, CTH_LAYER, &CTH_COLORMAP_M, &d, lat, lon);
            let cwp = sample_value(cache_dir, CWP_LAYER, &CWP_COLORMAP, &d, lat, lon);
            let reff = sample_value(cache_dir, REFF_LAYER, &REFF_COLORMAP, &d, lat, lon);
            SatelliteCloud {
                cot,
                cloud_top_m: cth,
                cwp_g_m2: cwp,
                r_eff_um: reff,
                age_days: age,
            }
        };
        match sample_layer(cache_dir, COT_LAYER, &COT_COLORMAP, &d, lat, lon) {
            Ok(LayerSample::Value(cot)) => return Some(attach(cot)),
            Ok(LayerSample::ClearPixel) => {
                // COT clear/missing at this pixel - the microphysics pair
                // can still measure the cloud (PCL pixels have CWP+r_eff
                // but no COT). Derive tau from them before declaring clear.
                let s = attach(0.0);
                if let Some(tau) = s.microphysics_tau() {
                    if tau > 0.1 {
                        return Some(SatelliteCloud { cot: tau, ..s });
                    }
                }
                return None; // genuinely clear at this pixel
            }
            Ok(LayerSample::EmptyTile) | Err(_) => continue, // not ingested; try older
        }
    }
    None
}

/// Observer + sunward-path composite sample ("2.5D"): the twilight shadow
/// path crosses the cloud field 50-300 km toward the sun azimuth.
pub fn sample_cloud_path(
    cache_dir: &Path,
    date: &str,
    lat: f64,
    lon: f64,
    sun_azimuth_deg: f64,
) -> SatelliteCloudPath {
    let observer = sample_cloud(cache_dir, date, lat, lon);

    let az = sun_azimuth_deg.to_radians();
    let mut cots = Vec::new();
    let mut cloudy = 0usize;
    let distances_km = [50.0, 100.0, 200.0, 300.0];
    for &dist in &distances_km {
        // local flat-earth offset (adequate at these distances)
        let dlat = dist * az.cos() / 111.0;
        let dlon = dist * az.sin() / (111.0 * lat.to_radians().cos().max(0.05));
        if let Some(s) = sample_cloud(cache_dir, date, lat + dlat, lon + dlon) {
            cots.push(s.cot);
            cloudy += 1;
        }
    }
    let n = distances_km.len();
    SatelliteCloudPath {
        observer,
        path_mean_cot: if cots.is_empty() {
            0.0
        } else {
            cots.iter().sum::<f64>() / cots.len() as f64
        },
        path_cloud_fraction: cloudy as f64 / n as f64,
        n_path_samples: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_math_known_points() {
        // Brondby 55.65N 12.41E: col = (192.41/9) = 21, row = (34.35/9) = 3
        let (row, col, px, py) = tile_for(55.653, 12.412);
        assert_eq!((row, col), (3, 21));
        assert!(px < TILE_PX && py < TILE_PX);
        // Mecca 21.42N 39.83E
        let (row, col, _, _) = tile_for(21.4225, 39.8262);
        assert_eq!((row, col), (7, 24));
        // date line wrap
        let (_, col, _, _) = tile_for(0.0, 179.9);
        assert_eq!(col, 39);
        let (_, col, _, _) = tile_for(0.0, -179.9);
        assert_eq!(col, 0);
    }

    #[test]
    fn lut_exact_and_near_match() {
        // first COT entry: (184,0,185) -> ~0.5
        let v = lut_lookup(&COT_COLORMAP, 184, 0, 185).unwrap();
        assert!(v > 0.0 && v < 1.0);
        // off-palette color -> None
        assert!(lut_lookup(&COT_COLORMAP, 10, 200, 10).is_none());
        // 1-step rounding tolerated
        assert!(lut_lookup(&COT_COLORMAP, 183, 0, 185).is_some());
    }

    #[test]
    fn cth_lut_in_meters() {
        let max = CTH_COLORMAP_M.iter().map(|e| e.3).fold(0.0f32, f32::max);
        assert!((10_000.0..=13_000.0).contains(&max), "CTH max = {max}");
    }

    #[test]
    fn date_minus_handles_month_and_year() {
        assert_eq!(date_minus("2026-01-01", 1).unwrap(), "2025-12-31");
        assert_eq!(date_minus("2026-03-01", 1).unwrap(), "2026-02-28");
        assert_eq!(date_minus("2024-03-01", 1).unwrap(), "2024-02-29");
    }
}
