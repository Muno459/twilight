#!/usr/bin/env python3
"""cloud3d on Meteosat SEVIRI via the EUMETSAT Data Store.

European/Indian-Ocean counterpart of cloud3d_profile.py (GOES). Downloads
the latest Meteosat-9 IODC HRSEVIRI L1.5 native scan (free EUMETSAT
account; credentials in ~/.eumdac/credentials), extracts an 11-channel
window around the observer in the NATIVE geostationary projection (the
model was trained on native MSG patches), runs the cloud3d IWC model and
emits the same JSON/PNG as the GOES sidecar.

SEVIRI is the instrument the model was TRAINED on — unlike GOES/AHI no
channel-mapping approximation is involved. Caveat for north-European
observers: from 45.5E the view zenith angle at e.g. Padborg is ~70 deg,
so native pixels are smeared to roughly 3.3 x 9.9 km; the reconstruction
is correspondingly coarser north-south.

Usage:
  python3 tools/cloud3d_seviri.py --lat 54.82 --lon 9.36 \
      --azimuth 45 --out profile.json --png padborg3d.png --place Padborg
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# satpy's dask threading segfaults on this Python; synchronous is plenty
# for one 128x128 window.
import dask

dask.config.set(scheduler="synchronous")

from cloud3d_profile import (
    MODEL_TOP_M,
    N_LEVELS,
    destination,
    iwc_from_output,
    normalize,
    render_png,
)

COLLECTION = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"
SAT_LON = 45.5  # Meteosat-9 IODC sub-satellite longitude
# The 11 narrow-band channels in wavelength-ascending order — exactly the
# training order; first three are solar/reflectance channels.
CHANNELS = ["VIS006", "VIS008", "IR_016", "IR_039", "WV_062", "WV_073",
            "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "data", "cloud3d", "seviri_cache")


def run_model_isolated(x_in, model_path):
    """Run the TorchScript model in a clean subprocess.

    torch and satpy cannot share this process: satpy's pykdtree and
    torch each bring their own OpenMP runtime on macOS and the model
    forward segfaults. Process isolation sidesteps it entirely.
    """
    with tempfile.TemporaryDirectory() as td:
        xin = os.path.join(td, "x.npy")
        yout = os.path.join(td, "y.npy")
        np.save(xin, x_in)
        code = (
            "import numpy as np, torch\n"
            f"x = np.load({xin!r})\n"
            f"m = torch.jit.load({os.path.abspath(model_path)!r}, map_location='cpu')\n"
            "m.eval()\n"
            "with torch.no_grad():\n"
            "    y = m(torch.from_numpy(x[None]))[0].numpy()\n"
            f"np.save({yout!r}, y)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
        return np.load(yout)


def get_token():
    path = os.path.expanduser("~/.eumdac/credentials")
    if not os.path.exists(path):
        raise SystemExit("no ~/.eumdac/credentials — register (free) at "
                         "eoportal.eumetsat.int and run "
                         "`eumdac set-credentials <key> <secret>`")
    import eumdac
    key, secret = open(path).read().strip().split(",")
    return eumdac.AccessToken((key, secret))


def fetch_latest_nat(token, before_iso=None):
    """Newest IODC scan (optionally at/before an ISO time) -> local .nat."""
    import eumdac
    import itertools

    ds = eumdac.DataStore(token)
    col = ds.get_collection(COLLECTION)
    kwargs = {}
    if before_iso:
        from datetime import datetime, timedelta
        end = datetime.fromisoformat(before_iso)
        kwargs = {"dtstart": end - timedelta(hours=6), "dtend": end}
    product = next(itertools.islice(col.search(**kwargs), 1))
    pid = str(product)

    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, pid + ".nat")
    if os.path.exists(local) and os.path.getsize(local) > 1e8:
        print(f"cloud3d-seviri: cached {pid}", file=sys.stderr)
        return local, pid, product.sensing_end

    nat_entry = next(e for e in product.entries if e.endswith(".nat"))
    print(f"cloud3d-seviri: downloading {pid} ({nat_entry})", file=sys.stderr)
    with product.open(entry=nat_entry) as fsrc, open(local + ".part", "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=1 << 20)
    os.rename(local + ".part", local)
    return local, pid, product.sensing_end


def lonlat_to_rowcol(area, lon, lat):
    """(row, col) of a lon/lat in the dataset grid, or None if outside."""
    try:
        col, row = area.get_array_indices_from_lonlat(lon, lat)
    except AttributeError:
        col, row = area.get_xy_from_lonlat(lon, lat)
    except ValueError:
        return None
    if np.ma.is_masked(col) or np.ma.is_masked(row):
        return None
    return int(row), int(col)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--azimuth", type=float, default=None,
                    help="sun azimuth deg (enables sunward path sampling)")
    ap.add_argument("--time", default=None,
                    help="UTC ISO time; default = newest available scan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default=None)
    ap.add_argument("--place", default=None)
    ap.add_argument("--model", default="data/cloud3d/iwc.jit.pt")
    ap.add_argument("--window", type=int, default=128)
    args = ap.parse_args()

    token = get_token()
    nat_path, pid, sensing_end = fetch_latest_nat(token, args.time)
    scan_time = sensing_end.strftime("%Y-%m-%dT%H:%MZ")

    # ── Load the 11 channels with physical calibration ──
    from satpy import Scene

    scn = Scene(reader="seviri_l1b_native", filenames=[nat_path])
    solar, thermal = CHANNELS[:3], CHANNELS[3:]
    scn.load(solar, calibration="reflectance")          # percent 0-100
    scn.load(thermal, calibration="brightness_temperature")  # K

    area = scn[CHANNELS[8]].attrs["area"]  # IR_108 full-disk area
    rc = lonlat_to_rowcol(area, args.lon, args.lat)
    if rc is None:
        print(json.dumps({"error": "below horizon",
                          "detail": f"{args.lat},{args.lon} not visible from {SAT_LON}E"}))
        return 2
    jc, ic = rc
    half = args.window // 2
    shape = scn[CHANNELS[8]].shape
    j0, j1 = max(0, jc - half), min(shape[0], jc + half)
    i0, i1 = max(0, ic - half), min(shape[1], ic + half)

    raw = np.empty((11, j1 - j0, i1 - i0), dtype=np.float32)
    for n, ch in enumerate(CHANNELS):
        raw[n] = scn[ch][j0:j1, i0:i1].values.astype(np.float32)

    x_in = normalize(raw)

    # ── Run the model (isolated process; see run_model_isolated) ──
    out = run_model_isolated(x_in, args.model)
    iwc = iwc_from_output(out)
    heights = np.linspace(MODEL_TOP_M, 0.0, N_LEVELS)

    # ── Aggregate (mirrors the GOES sidecar) ──
    h_, w_ = iwc.shape[1], iwc.shape[2]
    jc_w, ic_w = jc - j0, ic - i0

    def col_mean(jj, ii, r):
        ja, jb = max(0, jj - r), min(h_, jj + r + 1)
        ia, ib = max(0, ii - r), min(w_, ii + r + 1)
        return iwc[:, ja:jb, ia:ib].mean(axis=(1, 2))

    center = col_mean(jc_w, ic_w, 1)
    window_mean = iwc.mean(axis=(1, 2))
    dz = MODEL_TOP_M / (N_LEVELS - 1)
    iwp = iwc.sum(axis=0) * dz
    cloud_fraction = float((iwp > 1.0).mean())

    def sample_at(km, az):
        la, lo = destination(args.lat, args.lon, az, km)
        rc2 = lonlat_to_rowcol(area, lo, la)
        if rc2 is None:
            return None
        jj, ii = rc2[0] - j0, rc2[1] - i0
        if 0 <= jj < h_ and 0 <= ii < w_:
            return jj, ii
        return None

    path, cols, kms = [], [], []
    if args.azimuth is not None:
        for km in (0.0, 50.0, 100.0, 200.0, 300.0):
            s = sample_at(km, args.azimuth)
            if s:
                path.append({"km": km, "iwc_g_m3": col_mean(s[0], s[1], 2).tolist()})
        for km in np.arange(0.0, 321.0, 8.0):
            s = sample_at(float(km), args.azimuth)
            if s:
                cols.append(iwc[:, s[0], s[1]])
                kms.append(km)

    result = {
        "satellite": "meteosat-9-iodc",
        "granule": pid,
        "time_utc": scan_time,
        "requested_utc": args.time or "latest",
        "model": args.model,
        "heights_m": heights.tolist(),
        "cloud_fraction": cloud_fraction,
        "profiles": {
            "center": center.tolist(),
            "window_mean": window_mean.tolist(),
            "path": path,
        },
        "iwc_units": "g/m3",
    }
    with open(args.out, "w") as f:
        json.dump(result, f)

    if args.png and cols:
        # Ground sampling at the observer from the area's neighbor pixels
        # (Euclidean ground distance of one grid step; the grid axes are
        # rotated vs north this far off-nadir, so each step moves in both
        # lon and lat).
        lon_e, lat_e = area.get_lonlat(jc, ic + 1)
        lon_s, lat_s = area.get_lonlat(jc + 1, ic)
        lon_0, lat_0 = area.get_lonlat(jc, ic)
        coslat = math.cos(math.radians(lat_0))
        km_e = math.hypot(111.32 * coslat * (lon_e - lon_0),
                          110.57 * (lat_e - lat_0))
        km_s = math.hypot(111.32 * coslat * (lon_s - lon_0),
                          110.57 * (lat_s - lat_0))
        # synthesize the fields render_png expects from the GOES namespace
        args.date = scan_time[:10]
        args.hour = int(scan_time[11:13]) + int(scan_time[14:16]) / 60.0
        render_png(args.png, iwc, heights, args, scan_time,
                   "meteosat-9 (IODC 45.5E)", SAT_LON, None, None,
                   jc_w, ic_w, np.stack(cols, axis=1), np.array(kms),
                   px_km=(km_e, km_s))

    print(f"cloud3d-seviri: {pid} scan {scan_time}", file=sys.stderr)
    print(f"cloud3d-seviri: cloud fraction {cloud_fraction:.2f}, "
          f"max IWC {float(iwc.max()):.4f} g/m^3, path samples {len(path)}",
          file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
