#!/usr/bin/env python3
"""Directional marine-boundary-layer haze field for sea-horizon sites.

A 1D aerosol column cannot represent an observer on a coastline: the
dawn path toward the sea crosses hundreds of kilometres of marine haze
that the landward directions never see. This builds that asymmetry as
a georeferenced gray extinction field (the same sidecar the 3D cloud
transport consumes): a boundary-layer slab over SEA cells only, zero
over land, with climatological marine optical depth.

Physical inputs, stated plainly:
- Marine boundary-layer AOD (550 nm): default 0.12, the maritime-clean
  OPAC climatology; override with a measured value when one exists.
- Scale height 700 m (marine aerosol boundary layer).
- Coastline: a coarse polyline mask around the site (adequate because
  the low-elevation dawn path integrates over 100+ km; the mask only
  decides WHICH azimuths carry the haze).
- Gray treatment: marine aerosol has a small Angstrom exponent
  (~0.2 to 0.5), so a gray slab is a stated first-order approximation.

The engine's gray field channel expects an IWC-like g/m^3 grid that the
loader converts to extinction via its cloud mass-extinction constant
(BETA_PER_GM3, twilight-weather cloud3d loader). We invert that
constant here so the SLAB's integrated optical depth equals the target
AOD exactly after conversion.

Usage (Tubruq):
  python3 tools/marine_boundary_layer.py --lat 32.083 --lon 23.96 \
      --aod 0.12 --out /tmp/tubruq_marine.bin
"""

import argparse
import json
import re
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def beta_per_gm3():
    """Read the loader's mass-extinction relation so tau comes out
    exact (single source of truth: the cloud3d module doc, which the
    loader's unit tests pin): beta [1/m] = IWC [g/m^3] * 0.0545."""
    src = (ROOT / "crates/twilight-weather/src/cloud3d.rs").read_text()
    m = re.search(r"beta \[1/m\] = IWC \[g/m\^3\] \* ([0-9.]+[0-9])", src)
    if not m:
        raise SystemExit("IWC->beta relation not found in cloud3d.rs")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--aod", type=float, default=0.12)
    ap.add_argument("--scale-height", type=float, default=700.0)
    ap.add_argument("--half-extent-km", type=float, default=220.0)
    ap.add_argument("--res-km", type=float, default=4.0)
    ap.add_argument("--coast-lat", type=float, default=None,
                    help="sea = north of this latitude (simple coast); "
                         "default site latitude + 0.02")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    coast = a.coast_lat if a.coast_lat is not None else a.lat + 0.02
    dlat = a.res_km / 111.0
    dlon = a.res_km / (111.0 * np.cos(np.radians(a.lat)))
    n = int(2 * a.half_extent_km / a.res_km)
    # Vertical: 8 levels, 0 to 2100 m, 300 m steps (top-down order).
    heights = np.arange(2100.0, -1.0, -300.0)  # 2100..0, 8 levels
    nz = len(heights)

    # Slab density profile: exp(-z/H), normalized so that the column
    # integral of (beta_per_gm3 * iwc) dz equals the target AOD.
    beta = beta_per_gm3()
    prof = np.exp(-heights / a.scale_height)  # top-down
    dz = 300.0
    col = (prof * dz).sum()
    iwc0 = a.aod / (beta * col)  # g/m^3 at z=0

    lat_n = a.lat + n // 2 * dlat  # north edge
    lon_w = a.lon - n // 2 * dlon
    iwc = np.zeros((nz, n, n), dtype=np.float32)
    for iy in range(n):  # rows north to south
        cell_lat = lat_n - iy * dlat
        if cell_lat > coast:  # sea
            for iz in range(nz):
                iwc[iz, iy, :] = iwc0 * prof[iz]

    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from cloud3d_common import write_field

    write_field(a.out, iwc, heights, lat_n - n * dlat, lon_w, dlat, dlon,
                "climatology", f"marine_boundary_layer aod={a.aod} "
                f"H={a.scale_height} coast_lat={coast}")
    sea_frac = (iwc[nz - 1] > 0).mean()
    print(f"field written: {n}x{n}x{nz}, sea fraction {sea_frac:.2f}, "
          f"iwc0 {iwc0:.4e} g/m3, target AOD {a.aod}")
    print(json.dumps({"out": a.out, "beta_per_gm3": beta}, indent=1))


if __name__ == "__main__":
    main()
