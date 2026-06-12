//! Error type for the skyglow satellite feeds (atlas + DNB).

use std::fmt;

/// Failure modes of the atlas/DNB tile feeds.
#[derive(Debug)]
pub enum SkyglowError {
    /// Transport-level failure: timeout, DNS, refused connection, broken
    /// stream. No HTTP status was received.
    Network(String),
    /// The server answered with a non-success HTTP status.
    Http { status: u16, url: String },
    /// The tile arrived but could not be decoded (gunzip, PNG, palette).
    Parse(String),
    /// The location is outside the feed's coverage.
    NotCovered(String),
    /// Local filesystem failure (cache read/write).
    Io(String),
}

impl SkyglowError {
    pub fn network(msg: impl Into<String>) -> Self {
        SkyglowError::Network(msg.into())
    }

    pub fn http(status: u16, url: impl Into<String>) -> Self {
        SkyglowError::Http {
            status,
            url: url.into(),
        }
    }

    pub fn parse(msg: impl Into<String>) -> Self {
        SkyglowError::Parse(msg.into())
    }

    pub fn not_covered(msg: impl Into<String>) -> Self {
        SkyglowError::NotCovered(msg.into())
    }

    pub fn io(msg: impl Into<String>) -> Self {
        SkyglowError::Io(msg.into())
    }

    /// Retry-worthy failures: transport errors and server-side (5xx)
    /// statuses. Client errors (4xx, e.g. a date outside the layer's time
    /// range), parse failures, and coverage gaps are permanent.
    pub fn is_transient(&self) -> bool {
        match self {
            SkyglowError::Network(_) => true,
            SkyglowError::Http { status, .. } => (500..=599).contains(status),
            _ => false,
        }
    }

    /// Classify a ureq transport error against the URL it was sent to.
    pub(crate) fn from_ureq(e: ureq::Error, url: &str) -> Self {
        match e {
            ureq::Error::StatusCode(status) => SkyglowError::http(status, url),
            other => SkyglowError::network(format!("{other} ({url})")),
        }
    }
}

impl fmt::Display for SkyglowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkyglowError::Network(msg) => write!(f, "network error: {msg}"),
            SkyglowError::Http { status, url } => write!(f, "HTTP {status} from {url}"),
            SkyglowError::Parse(msg) => write!(f, "parse error: {msg}"),
            SkyglowError::NotCovered(msg) => write!(f, "not covered: {msg}"),
            SkyglowError::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for SkyglowError {}

impl From<std::io::Error> for SkyglowError {
    fn from(e: std::io::Error) -> Self {
        SkyglowError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_classification() {
        assert!(SkyglowError::network("timeout").is_transient());
        assert!(SkyglowError::http(502, "u").is_transient());
        assert!(!SkyglowError::http(400, "u").is_transient());
        assert!(!SkyglowError::parse("bad gzip").is_transient());
        assert!(!SkyglowError::not_covered("lat 80").is_transient());
        assert!(!SkyglowError::io("disk").is_transient());
    }
}
