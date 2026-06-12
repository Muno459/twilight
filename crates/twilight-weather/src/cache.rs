//! Disk-cache helpers for downloaded tiles.
//!
//! Shared by the GIBS satellite cache (data/satellite) and the skyglow
//! atlas/DNB cache (data/skyglow). Writes are atomic (.part then rename)
//! so a killed process never leaves a truncated tile that later reads as
//! valid; dated tiles are pruned after 14 days so the caches do not grow
//! without bound.

use std::path::Path;
use std::time::Duration;

/// Dated satellite tiles go stale quickly (each run fetches its own
/// dates); 14 days comfortably covers every date-fallback window in the
/// feeds while keeping the cache bounded.
const TILE_MAX_AGE: Duration = Duration::from_secs(14 * 24 * 60 * 60);

/// Atomically write a cache file: write `<path>.part`, then rename.
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".part");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

/// Best-effort eviction: delete regular files in `dir` whose name starts
/// with `prefix` and whose mtime is older than 14 days. An empty prefix
/// matches everything. Errors are ignored (a failed prune must never fail
/// the fetch that triggered it).
pub fn prune_stale(dir: &Path, prefix: &str) {
    prune_older_than(dir, prefix, TILE_MAX_AGE);
}

fn prune_older_than(dir: &Path, prefix: &str, max_age: Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(prefix) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let stale = meta
            .modified()
            .ok()
            .and_then(|m| now.duration_since(m).ok())
            .map(|age| age > max_age)
            .unwrap_or(false);
        if stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_atomic_leaves_no_part_file() {
        let dir = std::env::temp_dir().join("twilight_weather_cache_test_atomic");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("tile.png");
        write_atomic(&path, b"bytes").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"bytes");
        assert!(!dir.join("tile.png.part").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn prune_removes_only_old_matching_files() {
        let dir = std::env::temp_dir().join("twilight_weather_cache_test_prune");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("dnb_old.png"), b"x").unwrap();
        std::fs::write(dir.join("dnb_new.png"), b"x").unwrap();
        std::fs::write(dir.join("atlas_old.dat"), b"x").unwrap();
        // Everything was just written: zero max age marks all as stale,
        // but only the prefixed files may go.
        prune_older_than(&dir, "dnb_", Duration::ZERO);
        assert!(!dir.join("dnb_old.png").exists());
        assert!(!dir.join("dnb_new.png").exists());
        assert!(dir.join("atlas_old.dat").exists());
        // A generous max age keeps fresh files.
        std::fs::write(dir.join("dnb_fresh.png"), b"x").unwrap();
        prune_older_than(&dir, "dnb_", Duration::from_secs(3600));
        assert!(dir.join("dnb_fresh.png").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn prune_missing_dir_is_noop() {
        prune_stale(Path::new("/nonexistent/twilight/cache"), "");
    }
}
