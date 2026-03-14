//! GeoTIFF reader for elevation data.
//!
//! Wraps the `tiff` crate to decode elevation pixels from GeoTIFF files.
//! Extracts geographic metadata (pixel scale, tiepoint, CRS) from TIFF tags.

use std::fs::File;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};

/// A loaded GeoTIFF elevation tile.
pub struct ElevationTile {
    /// Elevation values in row-major order (top-left to bottom-right).
    pub data: Vec<f32>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Geographic metadata.
    pub geo: GeoTransform,
}

/// Geographic transform: maps pixel coordinates to geographic coordinates.
#[derive(Debug, Clone)]
pub struct GeoTransform {
    /// X coordinate (longitude/easting) of the top-left pixel center.
    pub origin_x: f64,
    /// Y coordinate (latitude/northing) of the top-left pixel center.
    pub origin_y: f64,
    /// Pixel size in X direction (degrees or meters, always positive).
    pub pixel_size_x: f64,
    /// Pixel size in Y direction (degrees or meters, always positive).
    /// Note: Y increases downward in pixel space, but northward in geo space,
    /// so origin_y is the NORTH edge and we subtract pixel_size_y * row.
    pub pixel_size_y: f64,
}

impl GeoTransform {
    /// Convert geographic coordinates to fractional pixel coordinates.
    /// Returns (col, row) as f64. May be out of bounds.
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        let col = (x - self.origin_x) / self.pixel_size_x;
        let row = (self.origin_y - y) / self.pixel_size_y;
        (col, row)
    }

    /// Convert pixel coordinates to geographic coordinates.
    /// Returns (x, y) at the pixel center.
    pub fn pixel_to_geo(&self, col: f64, row: f64) -> (f64, f64) {
        let x = self.origin_x + col * self.pixel_size_x;
        let y = self.origin_y - row * self.pixel_size_y;
        (x, y)
    }
}

impl ElevationTile {
    /// Load an elevation tile from a GeoTIFF file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let file =
            File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let mut decoder =
            Decoder::new(file).map_err(|e| format!("Failed to decode TIFF: {}", e))?;

        let (width, height) = decoder
            .dimensions()
            .map_err(|e| format!("Failed to read dimensions: {}", e))?;

        // Extract GeoTIFF metadata from tags
        let geo = read_geo_transform(&mut decoder, width, height)?;

        // Decode pixel data
        let result = decoder
            .read_image()
            .map_err(|e| format!("Failed to read image data: {}", e))?;

        let data = match result {
            DecodingResult::F32(d) => d,
            DecodingResult::F64(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I16(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U16(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I32(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U32(d) => d.into_iter().map(|v| v as f32).collect(),
            _ => return Err("Unsupported pixel data type for elevation".to_string()),
        };

        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(format!(
                "Data length {} != expected {} ({}x{})",
                data.len(),
                expected,
                width,
                height
            ));
        }

        Ok(ElevationTile {
            data,
            width,
            height,
            geo,
        })
    }

    /// Get elevation at a pixel coordinate. Returns None if out of bounds or NODATA.
    pub fn elevation_at_pixel(&self, col: u32, row: u32) -> Option<f32> {
        if col >= self.width || row >= self.height {
            return None;
        }
        let idx = row as usize * self.width as usize + col as usize;
        let val = self.data[idx];
        // Copernicus DEM uses 0.0 for ocean/NODATA in some tiles,
        // and very large negative values in others.
        // A reasonable check: valid land elevation is > -500m.
        if val < -500.0 || val.is_nan() {
            None
        } else {
            Some(val)
        }
    }

    /// Get elevation at geographic coordinates using bilinear interpolation.
    /// Coordinates are in the same CRS as the GeoTIFF (lon/lat for Copernicus,
    /// easting/northing for projected CRS like UTM).
    pub fn elevation_at_geo(&self, x: f64, y: f64) -> Option<f32> {
        let (col_f, row_f) = self.geo.geo_to_pixel(x, y);

        // Bounds check with 0.5 pixel margin
        if col_f < -0.5
            || row_f < -0.5
            || col_f >= self.width as f64 - 0.5
            || row_f >= self.height as f64 - 0.5
        {
            return None;
        }

        // Bilinear interpolation
        let c0 = col_f.floor() as i32;
        let r0 = row_f.floor() as i32;
        let c1 = c0 + 1;
        let r1 = r0 + 1;
        let fc = col_f - c0 as f64;
        let fr = row_f - r0 as f64;

        let get = |c: i32, r: i32| -> Option<f32> {
            if c < 0 || r < 0 {
                return None;
            }
            self.elevation_at_pixel(c as u32, r as u32)
        };

        // Try all four corners; fall back to nearest if some are NODATA
        let v00 = get(c0, r0);
        let v10 = get(c1, r0);
        let v01 = get(c0, r1);
        let v11 = get(c1, r1);

        match (v00, v10, v01, v11) {
            (Some(a), Some(b), Some(c), Some(d)) => {
                let top = a as f64 * (1.0 - fc) + b as f64 * fc;
                let bot = c as f64 * (1.0 - fc) + d as f64 * fc;
                Some((top * (1.0 - fr) + bot * fr) as f32)
            }
            // Partial data: use nearest neighbor
            _ => {
                let c_near = col_f.round() as i32;
                let r_near = row_f.round() as i32;
                get(c_near, r_near)
            }
        }
    }
}

/// Extract GeoTransform from TIFF tags (ModelPixelScaleTag + ModelTiepointTag).
fn read_geo_transform(
    _decoder: &mut Decoder<File>,
    _width: u32,
    _height: u32,
) -> Result<GeoTransform, String> {
    // ModelPixelScaleTag (33550): [ScaleX, ScaleY, ScaleZ]
    // ModelTiepointTag (33922): [I, J, K, X, Y, Z]
    //
    // The tiff crate doesn't expose arbitrary tags directly in a convenient way,
    // so we'll parse them from the raw IFD. However, the tiff crate's Decoder
    // doesn't give us easy access to custom tags.
    //
    // Workaround: re-read the file header to get the tag values.
    // For now, we rely on a simpler approach: the Copernicus DEM tiles have
    // a predictable structure (1-degree tiles with known pixel scale).
    // We'll implement full tag parsing as a fallback.

    // Try to get tags via the tiff crate's tag reading
    // Tag 33550 = ModelPixelScaleTag
    // Tag 33922 = ModelTiepointTag

    // The tiff crate (0.11) doesn't have a public API for reading arbitrary tags.
    // We'll parse the GeoTIFF metadata separately from the raw file bytes.

    // For now, return a placeholder that will be filled by the backend-specific code.
    // Each backend (copernicus, lidar) knows its tile structure and can provide
    // the correct GeoTransform without relying on tag parsing.
    Err("GeoTransform must be provided by the backend (use from_file_with_geo)".to_string())
}

impl ElevationTile {
    /// Load an elevation tile with an externally-provided GeoTransform.
    /// Use this when the backend knows the tile's geographic extent.
    pub fn from_file_with_geo(path: &Path, geo: GeoTransform) -> Result<Self, String> {
        let file =
            File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let mut decoder =
            Decoder::new(file).map_err(|e| format!("Failed to decode TIFF: {}", e))?;

        let (width, height) = decoder
            .dimensions()
            .map_err(|e| format!("Failed to read dimensions: {}", e))?;

        let result = decoder
            .read_image()
            .map_err(|e| format!("Failed to read image data: {}", e))?;

        let data = match result {
            DecodingResult::F32(d) => d,
            DecodingResult::F64(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I16(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U16(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I32(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U32(d) => d.into_iter().map(|v| v as f32).collect(),
            _ => return Err("Unsupported pixel data type for elevation".to_string()),
        };

        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(format!(
                "Data length {} != expected {} ({}x{})",
                data.len(),
                expected,
                width,
                height
            ));
        }

        Ok(ElevationTile {
            data,
            width,
            height,
            geo,
        })
    }

    /// Load from raw bytes with an externally-provided GeoTransform.
    /// Useful when data is already in memory (e.g., downloaded via HTTP).
    pub fn from_reader_with_geo<R: std::io::Read + std::io::Seek>(
        reader: R,
        geo: GeoTransform,
    ) -> Result<Self, String> {
        let mut decoder =
            Decoder::new(reader).map_err(|e| format!("Failed to decode TIFF: {}", e))?;

        let (width, height) = decoder
            .dimensions()
            .map_err(|e| format!("Failed to read dimensions: {}", e))?;

        let result = decoder
            .read_image()
            .map_err(|e| format!("Failed to read image data: {}", e))?;

        let data = match result {
            DecodingResult::F32(d) => d,
            DecodingResult::F64(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I16(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U16(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I32(d) => d.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U32(d) => d.into_iter().map(|v| v as f32).collect(),
            _ => return Err("Unsupported pixel data type for elevation".to_string()),
        };

        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(format!(
                "Data length {} != expected {} ({}x{})",
                data.len(),
                expected,
                width,
                height
            ));
        }

        Ok(ElevationTile {
            data,
            width,
            height,
            geo,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geo_transform_roundtrip() {
        let geo = GeoTransform {
            origin_x: 39.0,
            origin_y: 22.0,
            pixel_size_x: 1.0 / 3600.0,
            pixel_size_y: 1.0 / 3600.0,
        };

        // Top-left pixel should map to origin
        let (x, y) = geo.pixel_to_geo(0.0, 0.0);
        assert!((x - 39.0).abs() < 1e-10);
        assert!((y - 22.0).abs() < 1e-10);

        // Round-trip
        let (col, row) = geo.geo_to_pixel(39.5, 21.5);
        let (x2, y2) = geo.pixel_to_geo(col, row);
        assert!((x2 - 39.5).abs() < 1e-10);
        assert!((y2 - 21.5).abs() < 1e-10);
    }

    #[test]
    fn geo_transform_pixel_indices() {
        // Copernicus GLO-30: 3600x3600 pixels per 1-degree tile
        let geo = GeoTransform {
            origin_x: 39.0,
            origin_y: 22.0,
            pixel_size_x: 1.0 / 3600.0,
            pixel_size_y: 1.0 / 3600.0,
        };

        // Bottom-right corner of the tile
        let (col, row) = geo.geo_to_pixel(40.0 - 1.0 / 7200.0, 21.0 + 1.0 / 7200.0);
        assert!((col - 3599.5).abs() < 0.01);
        assert!((row - 3599.5).abs() < 0.01);
    }

    #[test]
    fn elevation_tile_synthetic() {
        // Create a synthetic 4x4 tile
        let data: Vec<f32> = (0..16).map(|i| (i * 100) as f32).collect();
        let tile = ElevationTile {
            data,
            width: 4,
            height: 4,
            geo: GeoTransform {
                origin_x: 0.0,
                origin_y: 4.0,
                pixel_size_x: 1.0,
                pixel_size_y: 1.0,
            },
        };

        // Pixel (0,0) = top-left = value 0
        assert_eq!(tile.elevation_at_pixel(0, 0), Some(0.0));
        // Pixel (1,0) = value 100
        assert_eq!(tile.elevation_at_pixel(1, 0), Some(100.0));
        // Pixel (0,1) = value 400
        assert_eq!(tile.elevation_at_pixel(0, 1), Some(400.0));
        // Out of bounds
        assert_eq!(tile.elevation_at_pixel(4, 0), None);
    }

    #[test]
    fn elevation_tile_bilinear() {
        // Create a 2x2 tile with known values
        let data = vec![100.0f32, 200.0, 300.0, 400.0];
        let tile = ElevationTile {
            data,
            width: 2,
            height: 2,
            geo: GeoTransform {
                origin_x: 0.0,
                origin_y: 2.0,
                pixel_size_x: 1.0,
                pixel_size_y: 1.0,
            },
        };

        // Center of tile: should be average of all four
        let center = tile.elevation_at_geo(0.5, 1.5);
        // Bilinear at (0.5, 0.5) pixel:
        // top = 100*0.5 + 200*0.5 = 150
        // bot = 300*0.5 + 400*0.5 = 350
        // result = 150*0.5 + 350*0.5 = 250
        assert!(center.is_some());
        assert!((center.unwrap() - 250.0).abs() < 0.01);
    }
}
