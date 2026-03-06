# twilight-terrain

Terrain masking for prayer time computation. Mountains, hills, and buildings that block the horizon shift when the sun becomes visible or when twilight fades. This crate computes horizon profiles from elevation data and adjusts the effective solar zenith angle.

## Data sources

**Copernicus GLO-30 DEM**. Global coverage at 30-meter resolution. Tiles are downloaded on demand from the Copernicus Prism API (free, no API key) and cached locally. Each tile is a GeoTIFF covering a 1-degree-by-1-degree cell. The crate downloads tiles for the observer's location and neighboring cells as needed.

**Danish SDFI LiDAR**. National LiDAR-derived surface model at 0.4-meter resolution. Accessed via the SDFI API. Optional backend for Denmark-specific high-resolution terrain.

## Modules

**`geotiff`**. Minimal GeoTIFF reader. Parses TIFF headers, strips/tiles, and geographic transform metadata (origin, pixel scale). Extracts elevation values as f64. Does not depend on GDAL or any C library.

**`copernicus`**. Copernicus GLO-30 tile management. Computes tile keys from coordinates, constructs download URLs, fetches tiles via `ureq`, parses GeoTIFF, and provides elevation lookup by latitude/longitude. Caches downloaded tiles to disk.

**`horizon`**. Horizon profile computation. For a given observer location, samples elevation at 360 azimuth bearings out to a configurable radius. At each azimuth, finds the maximum elevation angle above the local horizon. Returns a `HorizonProfile` (array of elevation angles indexed by azimuth). Computes the effective SZA adjustment: if the horizon is elevated by `h` degrees at the sun's azimuth, the sun rises later and sets earlier by the equivalent angular shift.

**`cache`**. Local tile cache with configurable directory. Avoids re-downloading tiles across runs.

**`projection`**. Geodetic coordinate utilities. Converts between lat/lon and the tile grid, handles tile boundary crossings.

**`lidar/`**. Danish SDFI LiDAR backend. Downloads and parses high-resolution surface model tiles.

## Tests

47 tests. GeoTIFF parsing (geo-transform, pixel indices, roundtrip), tile key computation (positive, negative, boundary coordinates), tile filename formatting, horizon profile computation (effective SZA adjustment, time shift), elevation lookup. 3 integration tests (Mecca Kaaba elevation, Mecca horizon profile, real tile loading) are ignored by default since they require network access.
