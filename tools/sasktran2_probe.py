#!/usr/bin/env python3
"""Feasibility probe: can SASKTRAN2 referee the deep-twilight regime?

Every external referee in the validation program so far (DISORT, MYSTIC)
comes from libRadtran, and both the deep-tier referee and the engine are
Monte Carlo. SASKTRAN2 (University of Saskatchewan) is independent on
both axes that matter:

  * different lineage, sharing no code with libRadtran; and
  * a DETERMINISTIC method - successive orders of scattering / discrete
    ordinates - so its error modes are unrelated to Monte Carlo variance.
    Agreement between a stochastic and a deterministic solver is a much
    stronger statement than agreement between two Monte Carlo codes.

It also offers what the regime requires: `GeometryType.Spherical` (not
merely pseudospherical), solar/LOS/multiple-scatter refraction switches,
and polarized output via `num_stokes`.

This script does not attempt a matched comparison. It answers the one
question that decides whether a matched comparison is worth building:
does SASKTRAN2 return finite, physically sensible zenith radiance at
solar zenith angles beyond 90 degrees, and how far past the terminator
does it stay usable?

Run with the probe environment:
    .venv-sasktran/Scripts/python.exe tools/sasktran2_probe.py
"""
import math
import sys

import numpy as np

try:
    import sasktran2 as sk
except ImportError:
    print("sasktran2 not installed in this interpreter", file=sys.stderr)
    raise SystemExit(1)


WAVELENGTHS = np.array([450.0, 550.0, 650.0])
SZA_SWEEP = [60.0, 80.0, 90.0, 95.0, 98.0, 101.0, 103.0, 105.0]


def run_sza(sza_deg, refraction=True, nstokes=1):
    """Zenith radiance from the ground at one solar zenith angle."""
    cos_sza = math.cos(math.radians(sza_deg))

    config = sk.Config()
    config.num_stokes = nstokes
    config.multiple_scatter_source = sk.MultipleScatterSource.SuccessiveOrders
    config.single_scatter_source = sk.SingleScatterSource.Exact
    config.num_streams = 16
    # Refraction is what makes past-terminator geometry meaningful.
    config.solar_refraction = refraction
    config.los_refraction = refraction
    config.multiple_scatter_refraction = refraction

    # 0-100 km, 1 km shells: the deep-twilight signal is carried high.
    altitude_grid = np.arange(0.0, 100001.0, 1000.0)
    geometry = sk.Geometry1D(
        cos_sza=cos_sza,
        solar_azimuth=0.0,
        earth_radius_m=6371000.0,
        altitude_grid_m=altitude_grid,
        interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.Spherical,
    )

    viewing = sk.ViewingGeometry()
    # Ground observer looking straight up.
    #
    # NOT GroundViewingSolar: that class is documented as "looking AT the
    # ground from angles defined at the ground location", i.e. a
    # downward-viewing geometry, and it segfaults when handed an upward
    # cos_viewing_zenith. SolarAnglesObserverLocation defines the angles
    # at the OBSERVER, which is what a ground-based up-looking site is.
    # cos_viewing_zenith = +1 is straight up; -1 returns identically zero.
    # relative_azimuth is in RADIANS.
    viewing.add_ray(
        sk.SolarAnglesObserverLocation(
            cos_sza=cos_sza,
            relative_azimuth=0.0,
            cos_viewing_zenith=1.0,
            observer_altitude_m=0.0,
        )
    )

    atmo = sk.Atmosphere(geometry, config, wavelengths_nm=WAVELENGTHS)
    sk.climatology.us76.add_us76_standard_atmosphere(atmo)
    atmo["rayleigh"] = sk.constituent.Rayleigh()

    engine = sk.Engine(config, geometry, viewing)
    return engine.calculate_radiance(atmo)


def main():
    print("SASKTRAN2 deep-twilight feasibility probe")
    print("spherical geometry, successive orders, refraction on, Rayleigh only")
    print()
    print(f"{'SZA':>7}" + "".join(f"{w:>14.0f} nm" for w in WAVELENGTHS))
    print("-" * (7 + 17 * len(WAVELENGTHS)))

    last = None
    for sza in SZA_SWEEP:
        try:
            out = run_sza(sza)
            rad = np.asarray(out["radiance"]).squeeze()
            vals = np.atleast_1d(rad)[: len(WAVELENGTHS)]
            cells = ""
            for v in vals:
                cells += f"{v:>17.4e}" if np.isfinite(v) else f"{'nonfinite':>17}"
            drop = ""
            if last is not None and np.all(np.isfinite(vals)) and np.all(last > 0):
                drop = f"   x{np.mean(vals / last):.3f} vs previous"
            print(f"{sza:>7.1f}{cells}{drop}")
            last = vals
        except Exception as exc:  # noqa: BLE001 - probe reports, does not raise
            msg = str(exc).splitlines()[0][:60]
            print(f"{sza:>7.1f}   FAILED: {msg}")
            last = None

    print()
    print("Reading it: radiance must stay finite and fall smoothly and")
    print("steeply past 90 deg. A flattening or a sign flip means the solver")
    print("has left its valid domain, and that SZA is the referee's limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
