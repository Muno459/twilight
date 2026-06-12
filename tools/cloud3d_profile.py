#!/usr/bin/env python3
"""cloud3d sidecar: 3D cloud vertical-profile reconstruction from live
geostationary imagery.

Runs the csaybar/cloud3d TorchScript model (SegFormer trained on the
Cloud3DTACO dataset, ESA FDL Earth Systems Lab 2025, arXiv:2511.04773) on
a window of near-real-time GOES ABI imagery and emits the 80-level
ice-water-content profile around an observer — real measured 3D cloud
structure for the twilight radiative-transfer engine.

Verified facts about the model (probed + dataset cross-check):
  input  : [B, 11, H, W] — the 11 SEVIRI narrow-band channels in
           wavelength-ascending order; on GOES ABI the nearest-wavelength
           set is C02,C03,C05 (solar) + C07,C08,C10,C11,C12,C13,C14,C16
           (thermal). Solar bands: TOA reflectance in PERCENT clipped to
           [0,100] then min-max to [-1,1]; thermal: BT in K clipped to
           [180,350] then min-max to [-1,1]; NaN -> 0 after normalization.
  output : [B, 80, H, W] — log-normalized IWC on the CloudSat bin grid,
           channel 0 = ~18,945 m (top), channel 79 = ~0 m, ~240 m bins.
           IWC [g/m^3] = 10 ** (((y+1)/2)*6 - 5); y=-1 means clear.

Data source: NOAA GOES on AWS Open Data, anonymous access. Lazy chunked
reads via s3fs/h5netcdf — only the chunks covering the requested window
are downloaded (tens of MB, not the 244 MB granule).

Coverage: GOES-19 (East, 75.2W) and GOES-18 (West, 137.0W) full disks.
Locations outside (Europe/Africa/Asia) need EUMETSAT SEVIRI/FCI
credentials — not implemented here; the engine falls back to the GIBS
MODIS COT+CTH route.

Usage:
  python3 tools/cloud3d_profile.py --lat 40.7 --lon -74.0 \
      --date 2026-06-12 --hour 8.5 --azimuth 62 --out profile.json
"""

import argparse
import json
import math
import re
import sys
import urllib.request

import numpy as np

# ── Model constants (verified against the Cloud3DTACO dataset) ──────
MODEL_TOP_M = 18945.0
N_LEVELS = 80
# GOES ABI channels matching the 11 SEVIRI narrow bands, wavelength order.
ABI_CHANNELS = ["C02", "C03", "C05", "C07", "C08", "C10", "C11", "C12", "C13", "C14", "C16"]
SOLAR = {"C02", "C03", "C05"}  # reflectance channels

SATELLITES = [
    # (bucket, sub-satellite longitude)
    ("noaa-goes19", -75.2),
    ("noaa-goes18", -137.0),
]
MAX_VIEW_DLON = 62.0  # beyond this the view zenith is too oblique to trust


def pick_satellite(lon):
    best = None
    for bucket, sat_lon in SATELLITES:
        d = abs((lon - sat_lon + 180.0) % 360.0 - 180.0)
        if d <= MAX_VIEW_DLON and (best is None or d < best[2]):
            best = (bucket, sat_lon, d)
    return best


def list_granules(bucket, date, hour):
    """List MCMIPF granule keys for the UTC hour via anonymous S3 REST."""
    doy = date.timetuple().tm_yday
    prefix = f"ABI-L2-MCMIPF/{date.year}/{doy:03d}/{int(hour):02d}/"
    url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
    with urllib.request.urlopen(url, timeout=30) as r:
        xml = r.read().decode()
    return re.findall(r"<Key>([^<]+\.nc)</Key>", xml)


def pick_granule(keys, hour):
    """Granule whose scan-start fractional hour is closest to `hour`."""
    best, best_d = None, 1e9
    for k in keys:
        m = re.search(r"_s\d{7}(\d{2})(\d{2})(\d{2})", k)
        if not m:
            continue
        h = int(m.group(1)) + int(m.group(2)) / 60.0 + int(m.group(3)) / 3600.0
        d = abs(h - hour % 24.0)
        if d < best_d:
            best, best_d = k, d
    return best


def latlon_to_scan(lat_deg, lon_deg, sat_lon_deg):
    """Geodetic -> ABI fixed-grid scan angles (GOES-R PUG L1b vol 3, 5.1.2.8.1)."""
    req = 6378137.0
    rpol = 6356752.31414
    h = 42164160.0  # satellite radius from Earth center
    e2 = 1.0 - (rpol / req) ** 2

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lam0 = math.radians(sat_lon_deg)

    phi_c = math.atan((rpol / req) ** 2 * math.tan(lat))
    rc = rpol / math.sqrt(1.0 - e2 * math.cos(phi_c) ** 2)
    sx = h - rc * math.cos(phi_c) * math.cos(lon - lam0)
    sy = -rc * math.cos(phi_c) * math.sin(lon - lam0)
    sz = rc * math.sin(phi_c)
    # visibility check
    if h * (h - sx) < sy * sy + (req / rpol) ** 2 * sz * sz:
        return None
    y = math.atan2(sz, sx)
    x = math.asin(-sy / math.sqrt(sx * sx + sy * sy + sz * sz))
    return x, y


def destination(lat_deg, lon_deg, azimuth_deg, dist_km):
    """Great-circle destination point."""
    r = 6371.0
    br = math.radians(azimuth_deg)
    d = dist_km / r
    la1 = math.radians(lat_deg)
    lo1 = math.radians(lon_deg)
    la2 = math.asin(
        math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(br)
    )
    lo2 = lo1 + math.atan2(
        math.sin(br) * math.sin(d) * math.cos(la1),
        math.cos(d) - math.sin(la1) * math.sin(la2),
    )
    return math.degrees(la2), math.degrees(lo2)


def normalize(stack):
    """[11,H,W] raw (refl % / BT K) -> [-1,1] with NaN->0, per the dataset."""
    out = np.empty_like(stack, dtype=np.float32)
    for i, ch in enumerate(ABI_CHANNELS):
        v = stack[i]
        if ch in SOLAR:
            v = np.clip(v, 0.0, 100.0)
            n = 2.0 * v / 100.0 - 1.0
        else:
            v = np.clip(v, 180.0, 350.0)
            n = 2.0 * (v - 180.0) / 170.0 - 1.0
        n[~np.isfinite(n)] = 0.0
        out[i] = n
    return out


def iwc_from_output(y):
    """Inverse transform: normalized output -> IWC g/m^3 (clear -> 0)."""
    y = np.clip(y, -1.0, 1.0)
    iwc = 10.0 ** (((y + 1.0) / 2.0) * 6.0 - 5.0)
    iwc[iwc < 2e-5] = 0.0  # y ~= -1 is the clear-sky fill (1e-5)
    return iwc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date", required=True, help="UTC date YYYY-MM-DD")
    ap.add_argument("--hour", type=float, required=True, help="UTC fractional hour")
    ap.add_argument("--azimuth", type=float, default=None,
                    help="sun azimuth deg (enables sunward path sampling)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="data/cloud3d/iwc.jit.pt")
    ap.add_argument("--window", type=int, default=128)
    args = ap.parse_args()

    from datetime import date as date_cls

    y_, m_, d_ = (int(v) for v in args.date.split("-"))
    date = date_cls(y_, m_, d_)

    sat = pick_satellite(args.lon)
    if sat is None:
        print(json.dumps({"error": "outside GOES coverage",
                          "detail": "lon %.1f not within %.0f deg of GOES-East/West; "
                                    "use the GIBS fallback" % (args.lon, MAX_VIEW_DLON)}))
        return 2
    bucket, sat_lon, _ = sat

    # Walk back from the requested hour to the most recent available scan
    # (the requested twilight hour is usually in the FUTURE when prayer
    # times are computed ahead of the prayer; GOES is observation-only, so
    # the latest cloud state is the best measured structure available).
    from datetime import timedelta

    keys, used = [], None
    when = date
    h = int(args.hour) % 24
    for _ in range(30):
        keys = list_granules(bucket, when, h)
        if keys:
            used = (when, h)
            break
        h -= 1
        if h < 0:
            h, when = 23, when - timedelta(days=1)
    if not keys:
        print(json.dumps({"error": "no granules", "detail": f"{bucket} {date} hour {args.hour}"}))
        return 2
    if used != (date, int(args.hour) % 24):
        print(f"cloud3d: requested {date} {args.hour:.1f}h not yet scanned; "
              f"using latest {used[0]} {used[1]:02d}h", file=sys.stderr)
    key = pick_granule(keys, args.hour)

    # ── Lazy open over anonymous S3 ──
    import s3fs
    import xarray as xr

    fs = s3fs.S3FileSystem(anon=True)
    # h5netcdf slicing is lazy: only the HDF5 chunks covering the window
    # are fetched over S3 range requests (tens of MB, not the full granule).
    ds = xr.open_dataset(fs.open(f"{bucket}/{key}"), engine="h5netcdf")

    scan = latlon_to_scan(args.lat, args.lon, sat_lon)
    if scan is None:
        print(json.dumps({"error": "below horizon", "detail": f"{bucket} cannot see this point"}))
        return 2
    cx, cy = scan
    xs = ds["x"].values  # radians, ascending
    ys = ds["y"].values  # radians, descending
    ic = int(np.argmin(np.abs(xs - cx)))
    jc = int(np.argmin(np.abs(ys - cy)))
    half = args.window // 2
    j0, j1 = max(0, jc - half), min(len(ys), jc + half)
    i0, i1 = max(0, ic - half), min(len(xs), ic + half)

    raw = np.empty((11, j1 - j0, i1 - i0), dtype=np.float32)
    for n, ch in enumerate(ABI_CHANNELS):
        v = ds[f"CMI_{ch}"][j0:j1, i0:i1].values.astype(np.float32)
        if ch in SOLAR:
            v = v * 100.0  # reflectance factor 0..1 -> percent
        raw[n] = v
    ds.close()

    x_in = normalize(raw)

    # ── Run the model ──
    import torch

    model = torch.jit.load(args.model, map_location="cpu")
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x_in[None]))[0].numpy()  # [80,H,W]
    iwc = iwc_from_output(out)
    heights = np.linspace(MODEL_TOP_M, 0.0, N_LEVELS)

    # ── Aggregate profiles ──
    h_, w_ = iwc.shape[1], iwc.shape[2]
    jc_w, ic_w = jc - j0, ic - i0

    def col_mean(jj, ii, r):
        ja, jb = max(0, jj - r), min(h_, jj + r + 1)
        ia, ib = max(0, ii - r), min(w_, ii + r + 1)
        return iwc[:, ja:jb, ia:ib].mean(axis=(1, 2))

    center = col_mean(jc_w, ic_w, 1)
    window_mean = iwc.mean(axis=(1, 2))
    # cloud fraction: columns with ice water path > 1 g/m^2
    dz = MODEL_TOP_M / (N_LEVELS - 1)
    iwp = iwc.sum(axis=0) * dz
    cloud_fraction = float((iwp > 1.0).mean())

    path = []
    if args.azimuth is not None:
        for km in (0.0, 50.0, 100.0, 200.0, 300.0):
            la, lo = destination(args.lat, args.lon, args.azimuth, km)
            s = latlon_to_scan(la, lo, sat_lon)
            if s is None:
                continue
            px, py = s
            ii = int(np.argmin(np.abs(xs[i0:i1] - px)))
            jj = int(np.argmin(np.abs(ys[j0:j1] - py)))
            # only accept points inside the window
            if 0 <= ii < w_ and 0 <= jj < h_:
                path.append({"km": km, "iwc_g_m3": col_mean(jj, ii, 2).tolist()})

    result = {
        "satellite": bucket,
        "granule": key,
        "time_utc": f"{args.date}T{int(args.hour):02d}:{int(args.hour % 1 * 60):02d}",
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
    print(f"cloud3d: {bucket} {key.split('/')[-1]}", file=sys.stderr)
    print(f"cloud3d: cloud fraction {cloud_fraction:.2f}, "
          f"max IWC {float(iwc.max()):.4f} g/m^3, path samples {len(path)}", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
