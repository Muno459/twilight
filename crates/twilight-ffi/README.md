# twilight-ffi

C-compatible FFI bindings for calling the twilight engine from iOS (Swift/ObjC), Android (Kotlin/Java via JNI), Flutter (Dart FFI), and any other language with C interop.

## Exports

```c
double twilight_solar_zenith(
    int year, int month, int day,
    int hour, int minute, int second,
    double timezone,
    double latitude, double longitude,
    double elevation
);
```

Returns the topocentric solar zenith angle in degrees, or -1.0 on error.

## Build

Compiles as both `cdylib` (shared library) and `staticlib` (static archive):

```bash
cargo build --release -p twilight-ffi

# Produces:
#   target/release/libtwilight_ffi.dylib  (macOS)
#   target/release/libtwilight_ffi.so     (Linux)
#   target/release/libtwilight_ffi.a      (static, all platforms)
```

For iOS, cross-compile with the appropriate target:

```bash
cargo build --release -p twilight-ffi --target aarch64-apple-ios
```

## Planned

- `twilight_compute_prayer_times`. full pipeline exposed as a C struct
- `twilight_spectral_radiance`. raw MCRT output for a given SZA
- Header file generation via `cbindgen`
