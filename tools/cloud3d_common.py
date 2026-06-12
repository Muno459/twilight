#!/usr/bin/env python3
"""Shared core of the cloud3d sidecars (cloud3d_profile.py = GOES,
cloud3d_seviri.py = Meteosat SEVIRI): profile aggregation, sunward
path/curtain sampling, result JSON assembly and the error protocol.

Row-orientation contract: every [80, H, W] IWC window handed to these
helpers has rows running NORTH to SOUTH (the GOES fixed-grid
convention). Natively south-up grids (satpy's SEVIRI reader) must be
flipped before aggregation; assert_rows_north_to_south is the cheap
hook callers use to verify the flip.

Error protocol (parsed by the Rust caller in twilight-cli - keep the
codes and exit statuses stable): a handled failure prints exactly ONE
JSON line to stdout,

    {"error": <code>, "detail": <human-readable detail>}

and exits with the code's class:

    2  coverage    - outside_coverage, no_granules, below_horizon
    3  environment - missing_deps (torch / satpy / model file absent)
    4  network     - network

Unhandled exceptions keep their traceback and exit nonzero. On success
the sidecar prints {"ok": true, "out": <profile json path>} and exits 0.
"""

import json
import sys

import numpy as np

# ── Model constants (verified against the Cloud3DTACO dataset) ──────
MODEL_TOP_M = 18945.0
N_LEVELS = 80

# Columns with ice water path above this are counted as cloudy [g/m^2].
CLOUDY_IWP_G_M2 = 1.0

# Sunward sampling grids: coarse distances for the JSON path profiles,
# fine grid for the figure curtain.
PATH_KM = (0.0, 50.0, 100.0, 200.0, 300.0)
CURTAIN_KM = np.arange(0.0, 321.0, 8.0)

EXIT_CODES = {
    "outside_coverage": 2,
    "no_granules": 2,
    "below_horizon": 2,
    "missing_deps": 3,
    "network": 4,
}


class SidecarError(Exception):
    """Handled failure carrying a protocol error code (module docstring)."""

    def __init__(self, code, detail):
        assert code in EXIT_CODES, code
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def run_main(main):
    """Entry-point wrapper enforcing the error protocol."""
    try:
        return main()
    except SidecarError as e:
        print(json.dumps({"error": e.code, "detail": e.detail}))
        return EXIT_CODES[e.code]


def ok(out_path):
    """Success line, the protocol counterpart of the error line."""
    print(json.dumps({"ok": True, "out": out_path}))
    return 0


def model_heights():
    """Bin-center heights [m], descending (index 0 = top ~18.9 km)."""
    return np.linspace(MODEL_TOP_M, 0.0, N_LEVELS)


def assert_rows_north_to_south(first_row_northing, last_row_northing):
    """Row-orientation contract hook. Callers pass any monotone northing
    proxy (latitude, GOES fixed-grid y) for the first and last window
    rows AFTER any flip; non-finite values (off-disk rows) are skipped."""
    if np.isfinite(first_row_northing) and np.isfinite(last_row_northing):
        assert first_row_northing >= last_row_northing, (
            "cloud3d window rows must run north to south "
            "(see the cloud3d_common docstring)")


def col_mean(iwc, jj, ii, r):
    """Mean IWC profile over a (2r+1)^2 px neighborhood, clipped to the
    window."""
    h_, w_ = iwc.shape[1], iwc.shape[2]
    ja, jb = max(0, jj - r), min(h_, jj + r + 1)
    ia, ib = max(0, ii - r), min(w_, ii + r + 1)
    return iwc[:, ja:jb, ia:ib].mean(axis=(1, 2))


def window_mean(iwc):
    """Mean IWC profile over the full window."""
    return iwc.mean(axis=(1, 2))


def cloud_fraction(iwc, heights):
    """Fraction of columns with ice water path > CLOUDY_IWP_G_M2."""
    dz = (heights[0] - heights[-1]) / (len(heights) - 1)
    iwp = iwc.sum(axis=0) * dz  # g/m^2
    return float((iwp > CLOUDY_IWP_G_M2).mean())


def sample_path(iwc, sampler, azimuth):
    """IWC column means at the PATH_KM distances toward `azimuth`.

    sampler(km, azimuth_deg) -> (jj, ii) window indices, or None for a
    point off the Earth disk or outside the window. Samplers must REJECT
    out-of-window points (return None), never clamp them to the edge.
    """
    path = []
    if azimuth is None:
        return path
    for km in PATH_KM:
        s = sampler(km, azimuth)
        if s is not None:
            path.append({"km": km,
                         "iwc_g_m3": col_mean(iwc, s[0], s[1], 2).tolist()})
    return path


def sample_curtain(iwc, sampler, azimuth):
    """Single-pixel IWC columns on the CURTAIN_KM grid toward `azimuth`
    (same sampler contract as sample_path); returns (cols, kms) lists."""
    cols, kms = [], []
    if azimuth is None:
        return cols, kms
    for km in CURTAIN_KM:
        s = sampler(float(km), azimuth)
        if s is not None:
            cols.append(iwc[:, s[0], s[1]])
            kms.append(float(km))
    return cols, kms


def write_result(out_path, satellite, granule, time_utc, requested_utc,
                 model, heights, cloud_frac, center, window_mean, path):
    """Assemble and write the profile JSON read by twilight-weather."""
    result = {
        "satellite": satellite,
        "granule": granule,
        "time_utc": time_utc,
        "requested_utc": requested_utc,
        "model": model,
        "heights_m": heights.tolist(),
        "cloud_fraction": cloud_frac,
        "profiles": {
            "center": center.tolist(),
            "window_mean": window_mean.tolist(),
            "path": path,
        },
        "iwc_units": "g/m3",
    }
    with open(out_path, "w") as f:
        json.dump(result, f)
    return result


def print_summary(tag, headline, iwc, cloud_frac, path):
    """Two-line stderr summary shared by the sidecars."""
    print(f"{tag}: {headline}", file=sys.stderr)
    print(f"{tag}: cloud fraction {cloud_frac:.2f}, "
          f"max IWC {float(iwc.max()):.4f} g/m^3, path samples {len(path)}",
          file=sys.stderr)
