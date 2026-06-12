//! Error type for the twilight-weather data feeds.

use std::fmt;

/// Failure modes of the weather/satellite/solar-flux feeds.
#[derive(Debug)]
pub enum WeatherError {
    /// Transport-level failure: timeout, DNS, refused connection, broken
    /// stream. No HTTP status was received.
    Network(String),
    /// The server answered with a non-success HTTP status.
    Http { status: u16, url: String },
    /// The response arrived but could not be decoded (JSON, PNG, ...).
    Parse(String),
    /// The feed answered cleanly but had no usable data for the request
    /// (missing hourly block, requested hour absent, no valid records).
    NoData(String),
    /// Local filesystem failure (cache read/write).
    Io(String),
}

impl WeatherError {
    pub fn network(msg: impl Into<String>) -> Self {
        WeatherError::Network(msg.into())
    }

    pub fn http(status: u16, url: impl Into<String>) -> Self {
        WeatherError::Http {
            status,
            url: url.into(),
        }
    }

    pub fn parse(msg: impl Into<String>) -> Self {
        WeatherError::Parse(msg.into())
    }

    pub fn no_data(msg: impl Into<String>) -> Self {
        WeatherError::NoData(msg.into())
    }

    pub fn io(msg: impl Into<String>) -> Self {
        WeatherError::Io(msg.into())
    }

    /// Retry-worthy failures: transport errors and server-side (5xx)
    /// statuses. Client errors (4xx), parse failures, and empty feeds are
    /// permanent for a given request.
    pub fn is_transient(&self) -> bool {
        match self {
            WeatherError::Network(_) => true,
            WeatherError::Http { status, .. } => (500..=599).contains(status),
            _ => false,
        }
    }

    /// Classify a ureq transport error against the URL it was sent to.
    pub(crate) fn from_ureq(e: ureq::Error, url: &str) -> Self {
        match e {
            ureq::Error::StatusCode(status) => WeatherError::http(status, url),
            other => WeatherError::network(format!("{other} ({url})")),
        }
    }
}

impl fmt::Display for WeatherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeatherError::Network(msg) => write!(f, "network error: {msg}"),
            WeatherError::Http { status, url } => write!(f, "HTTP {status} from {url}"),
            WeatherError::Parse(msg) => write!(f, "parse error: {msg}"),
            WeatherError::NoData(msg) => write!(f, "no data: {msg}"),
            WeatherError::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for WeatherError {}

impl From<std::io::Error> for WeatherError {
    fn from(e: std::io::Error) -> Self {
        WeatherError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for WeatherError {
    fn from(e: serde_json::Error) -> Self {
        WeatherError::Parse(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_classification() {
        assert!(WeatherError::network("timeout").is_transient());
        assert!(WeatherError::http(503, "u").is_transient());
        assert!(!WeatherError::http(404, "u").is_transient());
        assert!(!WeatherError::parse("bad json").is_transient());
        assert!(!WeatherError::no_data("empty").is_transient());
        assert!(!WeatherError::io("disk").is_transient());
    }

    #[test]
    fn display_includes_status_and_url() {
        let e = WeatherError::http(429, "https://example.org/x");
        let s = e.to_string();
        assert!(s.contains("429") && s.contains("example.org"));
    }
}
