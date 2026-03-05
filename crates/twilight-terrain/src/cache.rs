//! Disk cache for downloaded DEM tiles.
//!
//! Tiles are stored in a configurable directory (default: `data/dem/`).
//! Uses atomic writes to prevent partial files from corrupted downloads.

use std::fs;
use std::path::{Path, PathBuf};

/// Ensure the cache directory exists. Creates it if needed.
pub fn ensure_dir(dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create cache dir {}: {}", dir.display(), e))?;
    }
    Ok(())
}

/// Check if a cached tile exists.
pub fn tile_exists(dir: &Path, filename: &str) -> bool {
    dir.join(filename).exists()
}

/// Get the full path for a cached tile.
pub fn tile_path(dir: &Path, filename: &str) -> PathBuf {
    dir.join(filename)
}

/// Write data to a cache file atomically.
/// Writes to a .tmp file first, then renames to prevent partial files.
pub fn write_atomic(dir: &Path, filename: &str, data: &[u8]) -> Result<PathBuf, String> {
    ensure_dir(dir)?;

    let final_path = dir.join(filename);
    let tmp_path = dir.join(format!("{}.tmp", filename));

    fs::write(&tmp_path, data)
        .map_err(|e| format!("Failed to write {}: {}", tmp_path.display(), e))?;

    fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("Failed to rename {}: {}", tmp_path.display(), e))?;

    Ok(final_path)
}

/// Download a URL to the cache directory. Returns the cached file path.
/// Skips download if the file already exists.
pub fn download_to_cache(dir: &Path, filename: &str, url: &str) -> Result<PathBuf, String> {
    let final_path = dir.join(filename);
    if final_path.exists() {
        return Ok(final_path);
    }

    ensure_dir(dir)?;

    eprintln!("Downloading DEM tile: {}", filename);
    eprintln!("  URL: {}", url);

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP request failed for {}: {}", url, e))?;

    let status = response.status();
    if status == 404 {
        return Err(format!(
            "DEM tile not found (404): {} -- likely ocean area",
            url
        ));
    }
    if status != 200 {
        return Err(format!("HTTP {} for {}", status, url));
    }

    // Read response body
    let content_length = response
        .header("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50_000_000); // default 50MB buffer

    let mut body = Vec::with_capacity(content_length);
    response
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|e| format!("Failed to read response body: {}", e))?;

    let size_mb = body.len() as f64 / 1_048_576.0;
    eprintln!("  Downloaded {:.1} MB", size_mb);

    write_atomic(dir, filename, &body)
}

/// Subdirectory within the cache for a specific source.
pub fn source_dir(base_dir: &Path, source: &str) -> PathBuf {
    base_dir.join(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn ensure_dir_creates() {
        let tmp = env::temp_dir().join("twilight_cache_test_ensure");
        let _ = fs::remove_dir_all(&tmp);
        assert!(!tmp.exists());
        ensure_dir(&tmp).unwrap();
        assert!(tmp.is_dir());
        fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn write_atomic_and_read() {
        let tmp = env::temp_dir().join("twilight_cache_test_atomic");
        let _ = fs::remove_dir_all(&tmp);
        ensure_dir(&tmp).unwrap();

        let data = b"hello elevation data";
        let path = write_atomic(&tmp, "test_tile.tif", data).unwrap();
        assert!(path.exists());
        assert_eq!(fs::read(&path).unwrap(), data);

        // Temp file should not exist
        assert!(!tmp.join("test_tile.tif.tmp").exists());

        fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn tile_exists_check() {
        let tmp = env::temp_dir().join("twilight_cache_test_exists");
        let _ = fs::remove_dir_all(&tmp);
        ensure_dir(&tmp).unwrap();

        assert!(!tile_exists(&tmp, "nonexistent.tif"));
        fs::write(tmp.join("exists.tif"), b"data").unwrap();
        assert!(tile_exists(&tmp, "exists.tif"));

        fs::remove_dir_all(&tmp).unwrap();
    }
}
