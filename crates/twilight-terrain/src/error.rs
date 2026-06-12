//! Error type for the terrain DEM backends.

use std::fmt;

/// Failure modes of DEM tile download and decoding.
#[derive(Debug)]
pub enum TerrainError {
    /// Transport-level failure: timeout, DNS, refused connection, broken
    /// stream. No HTTP status was received.
    Network(String),
    /// The server answered with a non-success HTTP status (other than the
    /// 404 coverage gap, which is [`TerrainError::NotCovered`]).
    Http { status: u16, url: String },
    /// The tile arrived but could not be decoded (TIFF structure, pixel
    /// type, geo metadata).
    Parse(String),
    /// No tile exists for the location (404 = ocean area for Copernicus).
    NotCovered(String),
    /// Local filesystem failure (cache read/write).
    Io(String),
}

impl TerrainError {
    pub fn network(msg: impl Into<String>) -> Self {
        TerrainError::Network(msg.into())
    }

    pub fn http(status: u16, url: impl Into<String>) -> Self {
        TerrainError::Http {
            status,
            url: url.into(),
        }
    }

    pub fn parse(msg: impl Into<String>) -> Self {
        TerrainError::Parse(msg.into())
    }

    pub fn not_covered(msg: impl Into<String>) -> Self {
        TerrainError::NotCovered(msg.into())
    }

    pub fn io(msg: impl Into<String>) -> Self {
        TerrainError::Io(msg.into())
    }
}

impl fmt::Display for TerrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerrainError::Network(msg) => write!(f, "network error: {msg}"),
            TerrainError::Http { status, url } => write!(f, "HTTP {status} from {url}"),
            TerrainError::Parse(msg) => write!(f, "parse error: {msg}"),
            TerrainError::NotCovered(msg) => write!(f, "not covered: {msg}"),
            TerrainError::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for TerrainError {}

impl From<std::io::Error> for TerrainError {
    fn from(e: std::io::Error) -> Self {
        TerrainError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_distinguishes_variants() {
        assert!(TerrainError::not_covered("ocean")
            .to_string()
            .contains("not covered"));
        assert!(TerrainError::http(503, "u").to_string().contains("503"));
        assert!(TerrainError::parse("tiff").to_string().contains("parse"));
    }
}
