//! C FFI bindings for iOS/Android/Flutter.

use twilight_solar::spa;

/// Compute solar zenith angle for a given location and time.
///
/// # Safety
/// All parameters are plain values, no pointers — always safe.
#[no_mangle]
pub extern "C" fn twilight_solar_zenith(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
    timezone: f64,
    latitude: f64,
    longitude: f64,
    elevation: f64,
) -> f64 {
    let input = spa::SpaInput {
        year,
        month,
        day,
        hour,
        minute,
        second,
        timezone,
        latitude,
        longitude,
        elevation,
        pressure: 1013.25,
        temperature: 15.0,
        delta_t: 69.184,
        slope: 0.0,
        azm_rotation: 0.0,
        atmos_refract: 0.5667,
    };

    match spa::solar_position(&input) {
        Ok(output) => output.zenith,
        Err(_) => -1.0,
    }
}
