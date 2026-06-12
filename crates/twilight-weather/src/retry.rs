//! Bounded retry for transient fetch failures.
//!
//! Shared by every HTTP feed in the workspace (Open-Meteo, GIBS tiles,
//! SWPC F10.7, skyglow atlas and DNB tiles): up to 3 attempts with 1 s and
//! 3 s backoff. Only failures the caller classifies as transient (transport
//! errors, HTTP 5xx) are retried; client errors (4xx) and parse failures
//! fail on the first attempt.

use std::time::Duration;

/// Sleep schedule between attempts; its length fixes the retry budget
/// (3 attempts total).
const BACKOFF: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(3)];

/// Run `attempt` up to 3 times, sleeping 1 s then 3 s between attempts,
/// retrying only errors for which `is_transient` returns true.
pub fn with_retries<T, E>(
    is_transient: impl Fn(&E) -> bool,
    attempt: impl FnMut() -> Result<T, E>,
) -> Result<T, E> {
    with_backoff(&BACKOFF, is_transient, attempt)
}

/// Retry with an explicit sleep schedule (`backoff.len() + 1` attempts).
/// Split out so tests can run without real sleeps.
fn with_backoff<T, E>(
    backoff: &[Duration],
    is_transient: impl Fn(&E) -> bool,
    mut attempt: impl FnMut() -> Result<T, E>,
) -> Result<T, E> {
    let mut tries = 0usize;
    loop {
        match attempt() {
            Ok(v) => return Ok(v),
            Err(e) if tries < backoff.len() && is_transient(&e) => {
                std::thread::sleep(backoff[tries]);
                tries += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_SLEEP: [Duration; 2] = [Duration::ZERO, Duration::ZERO];

    #[test]
    fn first_success_no_retry() {
        let mut calls = 0;
        let r: Result<i32, ()> = with_backoff(
            &NO_SLEEP,
            |_| true,
            || {
                calls += 1;
                Ok(7)
            },
        );
        assert_eq!(r.unwrap(), 7);
        assert_eq!(calls, 1);
    }

    #[test]
    fn transient_errors_retried_up_to_three_attempts() {
        let mut calls = 0;
        let r: Result<i32, &str> = with_backoff(
            &NO_SLEEP,
            |_| true,
            || {
                calls += 1;
                Err("flaky")
            },
        );
        assert!(r.is_err());
        assert_eq!(calls, 3);
    }

    #[test]
    fn transient_then_success() {
        let mut calls = 0;
        let r: Result<i32, &str> = with_backoff(
            &NO_SLEEP,
            |_| true,
            || {
                calls += 1;
                if calls < 3 {
                    Err("flaky")
                } else {
                    Ok(42)
                }
            },
        );
        assert_eq!(r.unwrap(), 42);
        assert_eq!(calls, 3);
    }

    #[test]
    fn permanent_error_fails_immediately() {
        let mut calls = 0;
        let r: Result<i32, &str> = with_backoff(
            &NO_SLEEP,
            |e| *e != "fatal",
            || {
                calls += 1;
                Err("fatal")
            },
        );
        assert!(r.is_err());
        assert_eq!(calls, 1);
    }
}
