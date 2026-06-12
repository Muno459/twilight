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
    """[11,H,W] raw (refl % / BT K) -> [-1,1] with NaN->0, per the dataset.

    Channel order is wavelength-ascending (SEVIRI convention), so the
    first three slots are always the solar/reflectance channels — true
    for ABI, AHI and SEVIRI alike.
    """
    out = np.empty_like(stack, dtype=np.float32)
    for i in range(stack.shape[0]):
        v = stack[i]
        if i < 3:  # solar: reflectance %
            v = np.clip(v, 0.0, 100.0)
            n = 2.0 * v / 100.0 - 1.0
        else:  # thermal: BT K
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


def render_png(png_path, iwc, heights, args, scan_time, bucket, sat_lon,
               xs_win, ys_win, jc_w, ic_w, curtain, curtain_km,
               px_km=(2.0, 2.0)):
    """Figure: column-IWP map + vertical curtain along the sun azimuth +
    observer profile. Approximate orientation: rows ~ N->S, cols ~ W->E
    (geostationary grids are slightly rotated away from nadir).
    px_km = (east-west, north-south) ground sampling at the observer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    dz = heights[0] / (len(heights) - 1)
    iwp = iwc.sum(axis=0) * dz  # g/m^2
    h_, w_ = iwp.shape
    pxx, pxy = px_km
    extent_km = [-(ic_w) * pxx, (w_ - ic_w) * pxx,
                 -(h_ - jc_w) * pxy, (jc_w) * pxy]  # observer at 0,0

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)

    # ── Panel 1: ice water path map ──
    ax1 = fig.add_subplot(gs[0, 0])
    iwp_show = np.where(iwp > 0.5, iwp, np.nan)
    im1 = ax1.imshow(iwp_show, extent=extent_km, cmap="viridis",
                     norm=LogNorm(vmin=1.0, vmax=max(iwp.max(), 10.0)),
                     interpolation="nearest")
    ax1.set_facecolor("#1a1a2e")
    ax1.plot(0, 0, "r*", markersize=16, markeredgecolor="white", label="observer")
    if args.azimuth is not None:
        az = math.radians(args.azimuth)
        ax1.annotate("", xy=(120 * math.sin(az), 120 * math.cos(az)),
                     xytext=(0, 0),
                     arrowprops=dict(color="orangered", width=2, headwidth=10))
        ax1.text(130 * math.sin(az), 130 * math.cos(az), "to sun",
                 color="orangered", fontsize=10, ha="center")
    ax1.set_xlabel("km east")
    ax1.set_ylabel("km north")
    ax1.set_title("Ice water path (cloud3d reconstruction)")
    fig.colorbar(im1, ax=ax1, label="IWP [g/m$^2$]", shrink=0.85)

    # ── Panel 2: vertical curtain along the sun azimuth ──
    ax2 = fig.add_subplot(gs[0, 1])
    cshow = np.where(curtain > 1e-4, curtain, np.nan)
    im2 = ax2.pcolormesh(curtain_km, heights / 1000.0, cshow,
                         cmap="magma", norm=LogNorm(vmin=1e-3, vmax=0.5))
    ax2.set_facecolor("#10101c")
    ax2.set_xlabel("distance toward the sun [km]")
    ax2.set_ylabel("altitude [km]")
    ax2.set_ylim(0, 16)
    ax2.set_title("3D cloud curtain along the twilight light path")
    fig.colorbar(im2, ax=ax2, label="IWC [g/m$^3$]", shrink=0.85)

    # ── Panel 3: observer + window-mean profiles ──
    ax3 = fig.add_subplot(gs[1, 0])
    r = 1
    ja, jb = max(0, jc_w - r), min(h_, jc_w + r + 1)
    ia, ib = max(0, ic_w - r), min(w_, ic_w + r + 1)
    center = iwc[:, ja:jb, ia:ib].mean(axis=(1, 2))
    wmean = iwc.mean(axis=(1, 2))
    ax3.plot(center, heights / 1000.0, "-", color="tab:red", label="observer column")
    ax3.plot(wmean, heights / 1000.0, "-", color="tab:blue", label="window mean")
    ax3.set_xlabel("IWC [g/m$^3$]")
    ax3.set_ylabel("altitude [km]")
    ax3.set_xscale("symlog", linthresh=1e-4)
    ax3.set_ylim(0, 16)
    ax3.grid(alpha=0.3)
    ax3.legend()
    ax3.set_title("Vertical ice-water profile (80 CloudSat levels)")

    # ── Panel 4: provenance ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    dzkm = dz / 1000.0
    txt = (
        f"model     csaybar/cloud3d iwc.jit (SegFormer,\n"
        f"          trained on CloudSat radar profiles)\n"
        f"input     11-channel {bucket} full-disk scan\n"
        f"scan      {scan_time}\n"
        f"requested {args.date} {args.hour:.2f}h UTC\n"
        f"window    {iwc.shape[2]}x{iwc.shape[1]} px @ {pxx:.1f}x{pxy:.1f} km\n"
        f"vertical  80 levels x {dzkm * 1000:.0f} m (0-18.9 km)\n"
        f"cloudy    {100.0 * float((iwp > 1.0).mean()):.0f}% of columns (IWP > 1 g/m$^2$)\n"
        f"max IWC   {float(iwc.max()):.3f} g/m$^3$"
    )
    ax4.text(0.02, 0.95, txt, family="monospace", fontsize=10.5,
             va="top", transform=ax4.transAxes)

    place = args.place or f"{args.lat:.2f}N {args.lon:.2f}E"
    fig.suptitle(f"cloud3d — satellite 3D cloud reconstruction over {place}",
                 fontsize=15, y=0.99)
    fig.savefig(png_path, dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"cloud3d: figure -> {png_path}", file=sys.stderr)


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
    ap.add_argument("--png", default=None,
                    help="also render a figure (IWP map + sunward curtain + profiles)")
    ap.add_argument("--place", default=None, help="location label for the figure title")
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

    if args.png:
        # Vertical curtain along the sun azimuth (8 km sampling, 0-320 km).
        az = args.azimuth if args.azimuth is not None else 270.0
        cols, kms = [], []
        for km in np.arange(0.0, 321.0, 8.0):
            la, lo = destination(args.lat, args.lon, az, float(km))
            s = latlon_to_scan(la, lo, sat_lon)
            if s is None:
                continue
            px, py = s
            ii = int(np.argmin(np.abs(xs[i0:i1] - px)))
            jj = int(np.argmin(np.abs(ys[j0:j1] - py)))
            if 0 <= ii < w_ and 0 <= jj < h_:
                cols.append(iwc[:, jj, ii])
                kms.append(km)

    # The ACTUAL scan time from the granule name (_sYYYYDDDHHMMSSt) — the
    # requested twilight hour is usually in the future; never label model
    # output with a time the satellite did not observe.
    mscan = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})", key)
    if mscan:
        from datetime import datetime, timedelta as _td
        d0 = datetime(int(mscan.group(1)), 1, 1) + _td(days=int(mscan.group(2)) - 1)
        scan_time = f"{d0:%Y-%m-%d}T{mscan.group(3)}:{mscan.group(4)}Z"
    else:
        scan_time = "unknown"

    result = {
        "satellite": bucket,
        "granule": key,
        "time_utc": scan_time,
        "requested_utc": f"{args.date}T{int(args.hour):02d}:{int(args.hour % 1 * 60):02d}",
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
        render_png(args.png, iwc, heights, args, scan_time, bucket, sat_lon,
                   None, None, jc_w, ic_w,
                   np.stack(cols, axis=1), np.array(kms))

    print(f"cloud3d: {bucket} {key.split('/')[-1]}", file=sys.stderr)
    print(f"cloud3d: cloud fraction {cloud_fraction:.2f}, "
          f"max IWC {float(iwc.max()):.4f} g/m^3, path samples {len(path)}", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
