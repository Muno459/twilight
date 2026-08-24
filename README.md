<p align="center">
  <img src="assets/banner.svg" alt="twilight" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/Muno459/twilight"><img src="https://img.shields.io/badge/rust-workspace-orange?style=flat-square" alt="rust"/></a>
  <a href="https://github.com/Muno459/twilight"><img src="https://img.shields.io/badge/GPU-Metal%20%2B%20wgpu%20(Vulkan)-blueviolet?style=flat-square" alt="gpu"/></a>
  <a href="#validation"><img src="https://img.shields.io/badge/validated-libRadtran%20%2B%20SHDOM%20%2B%20field%20campaigns-green?style=flat-square" alt="validation"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="license"/></a>
</p>

<h3 align="center">A Monte Carlo radiative-transfer simulator of the twilight sky,<br/>with a validated model of human dawn and dusk visibility.</h3>
<p align="center">Its flagship application: Fajr and Isha computed the way the Quran defines them, not from a fixed angle.</p>

<br/>

> *"...and eat and drink until the **white thread** of dawn becomes **distinct to you** from the **black thread** of night..."* (al-Baqarah 2:187)

## What this is

`twilight` is two layers, cleanly separated in the crate graph.

**The simulator** computes absolute spectral sky radiance through deep
twilight: backward Monte Carlo transport through a 56-shell spherical
refracting atmosphere to 150 km, at 41 wavelengths with full Stokes
polarization, with measured 3D cloud fields, aerosols, five-species
molecular absorption, and every celestial and artificial background light
source. It is validated against DISORT, MYSTIC, and SHDOM (144/144 slab
points, absolute scale to 1 to 2.5 percent, cloud decks refereed to solar
zenith angle 103 with 1e9-photon references) and against measured twilight
skies. The same engine answers general questions: twilight brightness
decay, zodiacal light visibility, artificial skyglow, high-latitude
persistent twilight.

**The application** models a dark-adapted human observer watching the
horizon band, and reports the moment the white thread of dawn becomes
distinct and spreads, exactly as the verse and the sunnah describe it,
along with the false dawn (the narrow "wolf's tail" that rises before the
true dawn) and both Isha definitions (the disappearance of the red and the
white twilight). Prayer apps disagree because they reduce "when does dawn
become visible?" to a fixed solar depression angle (MWL 18 degrees, Egypt
15, Umm al-Qura offsets); the ayah describes a visual detection event, and
a detection event can be computed.

The sky it simulates is tonight's sky, not a textbook one: the forecast at
the prayer hour, measured aerosol optical depth, satellite-measured cloud
cover and city light, the real moon from the JPL ephemeris, and an **AI 3D
cloud reconstruction**: a neural network trained on spaceborne cloud radar
rebuilds the actual three-dimensional cloud field over your head from the
latest geostationary satellite scan, so the light transport sees clouds at
their true heights and thicknesses, not a guessed average. And because the
inputs are measured, the output carries an honest uncertainty: every
computed time is reported with a plus-minus band propagated from the Monte
Carlo noise, the skyglow atlas calibration, the street-light duty cycle,
and the aerosol product.

## Quick start

```bash
cargo build --release

# Clear-sky prayer times, anywhere on Earth. The timezone (including
# whether DST is active on that date) is determined automatically from
# the coordinates via the IANA database; --tz overrides if needed.
./target/release/twilight-cli pray --lat 21.4225 --lon 39.8262 --date 2026-06-13

# Production: live weather and aerosols at the prayer hour, satellite
# clouds, measured skyglow, JPL ephemeris, multi-scatter transport
./target/release/twilight-cli pray --lat 54.82 --lon 9.36 --date 2026-06-13 \
  --weather --skyglow --de440 data/de440.bsp --scattering hybrid
```

`--de440` is optional and the file is not in the repo (it is too large for
git). Fetch it once from NAIF if you want the JPL ephemeris instead of the
NREL SPA fallback; every command below works without the flag.

```bash
curl -o data/de440.bsp https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp
```

Output (Mecca, clear sky):

```
  Fajr (khayt al-abyad): 04:29:51 ±0.4min  (SZA 104.71°, depression 14.71°)
    └ white thread distinct from black + lateral spread (2:187, mustatir)
    └ false dawn (al-fajr al-kadhib) visible from 04:28:30 - do not pray Fajr yet
  Isha (shafaq ahmar):   20:12:59 ±5.3min  (SZA 104.96°, depression 14.96°)
    └ red band no longer distinct - Shafi'i/Maliki/Hanbali (primary)
  Isha (shafaq abyad):   20:25:13 ±0.5min  (SZA 107.33°, depression 17.33°)
    └ white band no longer distinct - Hanafi
```

Conventional fixed-angle times and a legacy absolute-threshold method are
printed below these for comparison.

## Validation

Two independent kinds of validation: the light transport is checked against
reference radiative-transfer codes, and the visibility criterion is checked
against published human dawn-observation campaigns. The full program lives
in `validation/` (one RESULTS document per campaign, each stating its
reproduction command).

**Criterion vs field campaigns** (clear sky, calibrated once at Mecca):

| Event | Engine | Independent measurement |
|---|---|---|
| Fajr sadiq (desert network) | **14.00 to 14.82°** across 10 sites | KACST 14.6 ± 0.3° · Hail 14.0 ± 0.3° · Aswan camera 14.90 ± 0.17° · every scored desert site matched within 0.45° |
| Isha al-abyad, Mecca | **17.33°** | SQM twilight-end 17.99 ± 0.16° (Niri et al. 2012, Sabah, Malaysia) · classical muwaqqit mode 17° |
| Isha al-ahmar, Mecca | **14.96°** | visual literature 12 to 15° ("colors gone before 15") |
| **Fajr, Birmingham UK** (out-of-sample, zero retuning, all 42 panel dates) | **mean residual +0.54°, RMS 0.95°** | **OpenFajr (CCD camera + 19-member scholar/observer panel); June trough 11.9 to 12.6° vs panel 12.3 to 12.9°** |

The Birmingham row matters most: the seasonal "summer relaxation" that UK
scholars apply as a hand-rule (14.5° in winter, about 12.5° in summer)
emerges from the physics twenty degrees of latitude from the desert
calibration campaigns with no additional tuning. Across the full campaign sweep the median absolute residual is
0.26° of depression over the 14 scored rows (eye 0.28°, instrument
0.26°). On the same 42-date Birmingham benchmark the fixed angles in
worldwide use miss by 1.59° RMS (ISNA 15°, itself undefined on 8 of
the 42 dates) and 4.41° RMS (MWL 18°, undefined on 15 of the 42);
definedness computed with the engine's own SPA ephemeris.

The criterion's one calibrated constant has itself been stress-tested:
inverting each desert campaign independently for its own implied value
gives a cluster with a multiplicative spread of only 1.31 (n = 8
sites, 25.8 to 32.1° N) and no latitude trend (r = +0.07): the
constant is the invariant that eight independent observation programs
agree on to within their own observational noise
(`validation/RESULTS_EDGE_FACTOR.md`). The production value (56) comes
from a full-campaign fit on the frozen final engine: an out-of-sample
residual minimum at f = 56.5 (RMS 0.133°), stable under leave-one-out
(mean 56.4, range 55.2 to 57.0), and sitting centrally in the 53 to 60
band that the per-site inversion implies (geometric mean 58). It
supersedes the earlier three-site cluster protocol, which selected 70
under the leverage of a single outlier (Hail, observed dawn 14.01°
against 14.5 to 14.8° everywhere else in the desert network) and
carries a 0.3 to 0.4° out-of-sample bias, two to three times the
per-site scatter. Every site outside the three calibration anchors
remains a genuine test.

**Transport vs reference codes** (details in `validation/RESULTS*.md`):

| Reference | Regime | Agreement |
|---|---|---|
| DISORT (pseudospherical, 16 streams) | SZA 60 to 90° | **2.6 to 3.5 % median** (shape) |
| DISORT, absolute scale (no normalization) | SZA 60 to 85° | **ratio 0.975 to 0.993** |
| MYSTIC (spherical Monte Carlo) | SZA 95 to 100° | **1 to 14 %**, backward mode, 1e8 photons |
| DISORT + MYSTIC, cloud slab (same delta-scaled HG problem) | tau* 1 / 3 / 10, SZA 30 to 60° | **144 of 144 points within 3 % + 2 SE**, both MC estimators |
| SHDOM, true-3D cloud cube | checkerboard + broken deck | **64/64 + 48/48 points in band** |
| MYSTIC ultra-deep, cloud decks at twilight | tau* 1 and 3, SZA 101 to 103, 3e8 to **1e9 photons** | **all sixteen cells PASS, 1d and 3D-field** (ratios 0.87 to 1.19); the two SZA 103 field cells closed by seed statistics alone (1024 and 512 seeds, field estimator unchanged) |
| Measured twilight skies (Patat, Koomen) | zenith decay into deep twilight | **1.2 to 5.5 %** per-magnitude decay error |

## Live data

Everything measurable is measured, fetched automatically, with documented
fallbacks:

| Quantity | Source | Cadence |
|---|---|---|
| Weather, cloud cover at the prayer hour | Open-Meteo forecast | hourly forecast |
| Aerosol optical depth (as excess over a measured baseline) | CAMS via Open-Meteo air quality, with per-value input sigma | hourly, archive since 2022-07 |
| Cloud optical thickness and top height | NASA GIBS, MODIS; sampled at the observer and 50 to 300 km along the sun azimuth, where the twilight light path actually crosses the cloud field | daily |
| Cloud water path and particle size (phase typing, gap filling) | NASA GIBS, MODIS microphysics | daily |
| 3D ice-cloud vertical profile | cloud3d model on GOES-19/18 (anonymous S3) or Meteosat-9 SEVIRI (EUMETSAT Data Store) | 10 to 15 min scans |
| Artificial skyglow | Lorenz 2024 propagated atlas, cross-checked by live VIIRS Black Marble nighttime lights; veil applied in mesopic photometry from the source spectrum (HPS vs LED) | atlas + daily |
| Solar activity (airglow driver) | NOAA SWPC F10.7 radio flux | daily measured |
| Sun and Moon | JPL DE440 ephemeris (exact lunar parallax, true phase angle, real distance); NREL SPA fallback | static file |
| Zodiacal light | Leinert 1998 Table 16, the measured grid, embedded and regenerable | embedded |
| Integrated starlight | Pioneer 10/11 sky maps (Toller 1981), digitized from the publisher scan | embedded |
| Moonlight | Krisciunas and Schaefer 1991, fed the DE440 lunar state | computed |
| Terrain horizon | Copernicus GLO-30 DEM | cached tiles |

## AI 3D cloud reconstruction

Clouds are the single largest physical influence on when dawn becomes
visible, and a 2D satellite picture cannot say how HIGH or how THICK they
are. So `twilight` runs the [cloud3d](https://huggingface.co/csaybar/cloud3d)
neural network (a SegFormer trained on CloudSat's spaceborne cloud radar,
ESA Cloud3DTACO dataset, arXiv:2511.04773) on the latest geostationary
satellite scan and reconstructs the full 3D cloud volume: an 80-level
vertical ice-water-content profile for every pixel, on a 240 m vertical
grid up to 19 km.

```bash
# Americas/Pacific: GOES-19/18, no account needed (anonymous AWS S3)
./target/release/twilight-cli pray ... --cloud3d auto

# Europe/Africa/Asia: Meteosat-9 SEVIRI via a free EUMETSAT account
pip install eumdac satpy torch
eumdac set-credentials <consumer-key> <consumer-secret>   # eoportal.eumetsat.int
python3 tools/cloud3d_seviri.py --lat 54.82 --lon 9.36 --azimuth 45 \
  --out profile.json --png3d clouds3d.png
./target/release/twilight-cli pray ... --cloud3d profile.json
```

The `--png3d` flag renders the reconstructed volume as a true 3D scene:
cloud isosurfaces colored by altitude, standing on the actual satellite
image as the ground plane. The engine samples the volume along the sun
azimuth (where the twilight light path actually crosses the cloud field),
collapses it into vertical layers at their measured heights, and rescales
the total amplitude to the MODIS-measured optical thickness when one is
available: structure from the AI reconstruction, amplitude from direct
measurement. SEVIRI is the instrument the model was trained on; GOES uses
the eleven nearest-wavelength channels. With `--cloud-field` the transport
path-traces the georeferenced voxel volume directly (explicit in-cloud
scattering on CPU and GPU alike).

### More options

```bash
twilight-cli pray ... --terrain               # Copernicus DEM horizon masking
twilight-cli pray ... --bortle 7              # manual skyglow (or --radiance)
twilight-cli pray ... --aerosol urban --cloud thin-cirrus
twilight-cli pray ... --photons 500           # more MC rays per LOS step
twilight-cli pray ... --cpu                   # skip the GPU
twilight-cli pray ... --fast                  # scalar mode, skip polarization
twilight-cli mcrt  --lat 21.42 --lon 39.83 --sza-start 90 --sza-end 108
twilight-cli solar --lat 21.42 --lon 39.83 --date 2026-06-15 --tz 3 --de440 data/de440.bsp
```

## How it works

### The criterion

For every scanned solar depression the engine simulates seven sky patches:
five across the dawn band at 3° altitude spanning ±18° of the solar azimuth,
plus dark reference patches at ±100°. The geometry comes from measurement:
the twilight arch sits at 2.66 ± 0.23° altitude (Aswan camera campaign), the
true-dawn band is roughly 30 to 40° wide at the moment of distinctness
(Ilyas; the Hail observers), and the false dawn's zodiacal wedge is about
20° wide at its base (Sultan).

Each patch's brightness is the sum of simulated twilight, the
direction-dependent celestial background (zodiacal light, starlight,
airglow, moonlight), and skyglow. Detection asks the question the ayah
asks: has the growth of each band patch above its own deep-night baseline
become distinct, as a Weber contrast against the reference-sky adaptation,
simultaneously across the full lateral extent? When only the central
patches pass, that is the false dawn; when the whole band passes, that is
Fajr sadiq. Isha mirrors the test as a disappearance, with the red channel
gated at the cone threshold (below about 10⁻³ cd/m² rods see no color, so
there is no red shafaq to lose).

A relative criterion has a quiet superpower: systematic errors that
multiply both patches (absolute radiometric scale, uniform cloud cover,
uniform skyglow) largely cancel in the ratio. It also needs no special
handling at high latitudes: the test is already relative to tonight's sky,
so brightness-based times exist wherever the sky physically brightens
(Padborg at 54.8° N in June: Fajr 02:55 inside persistent twilight,
confirmed by direct observation).

One calibration layer is stated openly. Laboratory psychophysics predicts
the eye should already see the dawn's photometric excess at depression 17
to 18°; every field campaign shows it does not, because the dawn at that
depth is a borderless gradient spread over tens of degrees (photometers
detect it, people do not). The factor bridging that gap is calibrated once
against the Mecca campaign cluster and then held fixed worldwide. Three
independent attacks on that constant (per-site inversion across eight
desert campaigns, a decomposition from the published psychophysics
literature, and a zodiacal-light cross-prediction it was never tuned on)
are documented in `validation/RESULTS_EDGE_FACTOR.md`; the Birmingham row
in the validation table remains the cleanest out-of-sample test.

Set `TWILIGHT_KHAYT_DEBUG=1` to dump the per-azimuth contrast-margin curves
of any run, including the per-crossing uncertainty split (Monte Carlo,
skyglow calibration, street-light duty cycle, aerosol input).

### The transport

The sky itself is computed by backward Monte Carlo radiative transfer
through a 56-shell spherical atmosphere reaching 150 km: Rayleigh
scattering, five-species molecular absorption (O3, NO2, O2, H2O, O4 CIA
from HITRAN/Serdyuchenko data), OPAC-style aerosols, measured 3D cloud
fields traversed by exact DDA voxel marching (with a gray combined channel
and per-shell majorants for forced flights), full Stokes polarization, and
Snell refraction at shell boundaries, at 41 wavelengths from 380 to 780 nm.

First-order scattering is evaluated as an exact deterministic integral.
The entire Monte Carlo budget goes to the multiple-scattering field, which
is all that exists at Fajr depth, where the line of sight lies fully inside
Earth's shadow. The estimator stack (next-event estimation, one-sample-MIS
seeding, forced collisions via truncated null-collision delta tracking,
VSPG collision-location importance, exponential transform, Dwivedi
directional MIS, weight windows with splitting, a forward-informed
importance map, bidirectional subpaths, hero-wavelength spectral MIS) is
exactly unbiased and held to that standard by an adversarial audit process,
bit-identity harnesses across refactors, and regression-gated parity tests
between four independent implementations (CPU scalar, CPU polarized, Metal,
WGSL).

A full computed day runs about 430 million photon chains: roughly 520 sky
patches (depression scan × view fan × independent seeds), each tracing
20,000 chains per wavelength across 41 wavelengths, with every scattering
event firing a refracted shadow ray toward the sun. On an Apple-Silicon
GPU this takes minutes; the same WGSL kernels run on any Vulkan, DX12, or
Metal adapter through the portable wgpu backend (headless Linux/NVIDIA
included).

## Architecture

The workspace boundary is the simulator/application boundary:

```
crates/                              THE SIMULATOR
  twilight-core       MC transport kernel (#![no_std], forbid(unsafe));
                      photon chains, 3D cloud DDA, importance machinery
  twilight-data       atmosphere builder: USSA-76 + thermosphere to 150 km,
                      prebaked cross-sections, aerosols, cloud optics
  twilight-solar      NREL SPA, pure-Rust JPL DE440 SPK reader, lunar
                      ephemeris, earth rotation
  twilight-gpu        Metal + portable wgpu (Vulkan/DX12/Metal) ports of
                      the full estimator, parity-gated against the CPU
  twilight-threshold  CIE mesopic/scotopic luminance, contrast thresholds,
                      night-sky background from measured tables

                                     THE APPLICATION
  twilight-cpu        pipeline: K-seed scans, the khayt fan, crossings,
                      uncertainty propagation
  twilight-weather    Open-Meteo, CAMS aerosols, GIBS satellite, cloud3d
  twilight-skyglow    Lorenz atlas, VIIRS Black Marble, Garstang model
  twilight-terrain    Copernicus DEM horizon profiles
  twilight-ffi        C FFI (minimal)
  twilight-cli        the twilight-cli binary

tools/
  validate_libradtran.py   DISORT/MYSTIC cross-validation harness (all tiers)
  criterion_edge_factor.py the edge-factor attack program
  cloud3d_profile.py       GOES to cloud3d 3D profiles (+ 3D renders)
  cloud3d_seviri.py        Meteosat SEVIRI to cloud3d (EUMETSAT Data Store)
  gen_*.py                 every embedded data table is regenerable from source
```

See `docs/ARCHITECTURE.md` for the layer contract and
`validation/README.md` for the map from every results table to the command
that regenerates it.

## Limitations

Treat computed times as **experimental for worship**; cross-check against
established local calendars. Known limits, stated plainly:

- **No field calibration of this engine's own output yet.** The criterion
  is calibrated against published campaigns. The measurement apparatus now
  ships with the engine (`twilight-cli sqm predict|compare` and the
  campaign protocol in docs/SQM_CAMPAIGN.md); what remains is a meter on a
  roof and clear nights.
- **The connection estimator extends to voxel fields.** The heavy-tail
  limitation on the deep cells was resolved by the bidirectional
  connection estimator for scattering orders three and above: light
  subpaths traced from the sunlit top of atmosphere form a vertex
  registry, every backward-chain collision beyond SZA 99 connects to one
  registry vertex, and each order combines the connection with its
  next-event counterpart through an exact per-path balance-heuristic MIS
  (both weights come from one shared function and sum to one). Under a
  3D field the light subpath's optical depth is exactly integrable by
  the same voxel traversal the chains use, so the extension carries no
  majorant and no new tail (`docs/FIELD_CONNECTIONS_PLAN.md`, merged
  with its G-FC gate ladder green). All sixteen deep cells PASS the
  MYSTIC referee (ratios 0.87 to 1.19): the refereed field rows are the
  earlier analog chain at 1024/512 seeds, and re-refereeing them with
  the connection estimator at 128 seeds reproduces the same means
  against the same references, with cross-machine agreement to three
  decimals (`validation/RESULTS_DEEP_HARDENING.md` section 7). A
  protocol note from that re-referee: the tier's 1d rows are scalar and
  its field rows polarized, and the polarized protocol reads about +10
  percent against the scalar referee on the 1d deck itself; within one
  protocol the representations agree to paired-test precision. The
  gate taxonomy in `validation/RESULTS_DEEP_REGIME.md` tracks every cell.
- **Zenith views over broken decks starve the importance sampler** (a
  documented residual with a false-tight error signature; slant views,
  which the khayt band uses, are unaffected).
- **GPU population control runs on a capped split stack.** The hybrid
  chain kernels carry the CPU's weight windows (splitting + Russian
  roulette on the altitude + CADIS importance) with a 4-slot per-thread
  stack against the CPU's 24, accepting the thread divergence (measured:
  no wall-clock cost; deep-field calls got faster from the RR kills).
  The GPU uses the heuristic window target only; the CPU keeps the
  forward-informed importance map and the full stack, so it remains the
  variance reference in the deepest cells.
- **Sea-horizon sites carry a partially closed offset** (Tubruq): a
  georeferenced marine boundary-layer haze slab over the sea cells along
  the dawn path (`tools/marine_boundary_layer.py`, climatological maritime
  AOD 0.12 at 550 nm, 700 m scale height) shifts the computed sea-horizon
  dawn by -0.54° averaged over three seasons, closing roughly half the
  +1.2° residual; the remaining ~+0.5° stays open (episodically thicker
  marine haze than climatology, or a different visual task over a
  featureless sea horizon).
- **Shafaq al-ahmar** rests on the weakest observational dataset (no
  color-resolved modern campaign exists); the evening calibration leans on
  instrumental rather than panel data.

## Reproduce the validation

```bash
# libRadtran cross-validation (needs a libRadtran build; see tool docstring)
export LIBRADTRAN_DIR=~/tools-build/libRadtran-2.0.6
python3 tools/validate_libradtran.py --tier 1a --shape-only
python3 tools/validate_libradtran.py --tier 1b --shape-only
python3 tools/validate_libradtran.py --tier deep     # 3e8 to 1e9 photon referee

# Edge-factor attack program (per-site inversion, zodiacal ladder)
python3 tools/criterion_edge_factor.py --analyze

# Field-campaign checks
./target/release/twilight-cli pray --lat 21.4225 --lon 39.8262 --date 2026-06-13 \
  --de440 data/de440.bsp --scattering hybrid   # Mecca: 14.71/17.33/14.96
./target/release/twilight-cli pray --lat 52.44 --lon=-1.93 --date 2026-06-13 \
  --de440 data/de440.bsp --scattering hybrid   # Birmingham June: 12.78

# Full test suite (both GPU backends and their parity gates)
cargo test --workspace --release
cargo test -p twilight-gpu --release --features "metal wgpu"
```

Every table in `validation/RESULTS*.md` states the exact command and gate
that produced it; `validation/README.md` is the index.

## Citing

If you use this simulator in academic work, please cite it via the
repository's `CITATION.cff` (GitHub renders a "Cite this repository"
button). A methods and validation paper is in preparation.

## License

MIT OR Apache-2.0, at your option.

ولله الحمد. For the ummah.
