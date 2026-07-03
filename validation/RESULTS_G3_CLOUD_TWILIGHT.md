# G3: Clouds at twilight geometry, external referee (2026-07-02)

Two campaigns extending the external cloud-transport validation from the
G2 daytime slab referee (validation/RESULTS.md) into the twilight regime
proper. Branch validation-campaigns on top of main 25c678a; referee is
the public libRadtran 2.0.6 build at ~/tools-build/libRadtran-2.0.6.
Runner: `tools/validate_libradtran.py --tier g3-cloud-twilight` and
`--tier g3-cube`. Raw decks and outputs: `validation/g3/` and
`validation/g3cube/`; full machine table
`validation/g3_cloud_twilight_results.csv`; campaign log
`validation/g3_campaign.log`.

## Campaign A: g3-cloud-twilight (the confidence-row upgrade)

Purpose: the "clouds at twilight geometry" row was previously anchored
only by internal gates plus a 2-point external arbitration (SZA 95/97,
one thin deck, vz 80, RESULTS.md "Deep-twilight arbitration"). This
campaign referees the uniform 1-2 km deck at SZA 95/97/99/101, zenith
view, 450/550/650 nm, tau* in {1, 3} for BOTH chain estimators and
tau* = 10 for the Multiple estimator, against spherical BACKWARD MYSTIC,
the only public referee that is valid past SZA ~95.

### Same-problem construction (identical to G2)

twilight's cloud channel is delta-Eddington scaled at build time, so the
libRadtran water cloud is configured with the SAME scaled triplet the
twilight medium actually carries (water preset ssa 0.999, g 0.85,
de-scale 0.2782225, ssa* 0.9964058, g* 0.4594595; `wc_properties hu` is
HG by construction, `wc_modify tau/ssa/gg set` are gray):

| case | twilight `--cloud-tau` (unscaled) | uvspec `wc_modify tau set` |
|-----:|----:|----:|
| tau* 1 | 3.594246 | 1.000000 |
| tau* 3 | 10.782737 | 3.000000 |
| tau* 10 | 35.942456 | 10.000000 |

Both codes: Rayleigh-only gas (`no_absorption mol` + `mol_abs_param crs`
vs `--rayleigh-only`), albedo 0.15, straight shadow rays (MYSTIC
mc_spherical does not refract; twilight `--no-refraction`), scalar
radiance (`--fast`), and the identical TSIS-1 10 nm solar table
(`validation/g2_solar_tsis_tw.dat`), so the comparison is absolute with
no shape normalization. twilight carries its 150 km atmosphere (the
100 km ceiling fix); afglus tops at 120 km, a residual geometry
difference that is far below the statistical bands through SZA 101
(shadow height at SZA 101 is ~60 km).

Referee deck (representative; all in `validation/g3/*/case.inp`):

```
atmosphere_file <libRadtran>/data/atmmod/afglus.dat
source solar validation/g2_solar_tsis_tw.dat per_nm
albedo 0.15
wavelength 550.0 550.0
mol_abs_param crs
no_absorption mol
wc_file 1D validation/g2_wc_slab.dat      # 1-2 km slab
wc_properties hu
wc_modify tau set 3.000000                # tau*
wc_modify ssa set 0.996406                # ssa*
wc_modify gg set 0.459459                 # g*
rte_solver montecarlo
mc_spherical 1D
mc_photons 100000000                      # 1e7 at SZA 95/97, 1e8 at 99/101
mc_backward
mc_vroom on
mc_std
sza 99.00
phi0 0.0
umu -1.000000                             # zenith sensor
phi 0.0
zout 0.0
output_user lambda uu
quiet
```

twilight side (per seed):

```
twilight-cli compare --sza 95,97,99,101 --view-zenith 0 --rel-azimuth 0
  --rayleigh-only --fast --no-refraction --scattering {hybrid|multiple}
  --photons {16000 | 500000 | 2000000} --seed-salt {1..6}
  --cloud-tau <tau*/0.2782225> --cloud-base-km 1 --cloud-top-km 2
  --cloud-ssa 0.999 --cloud-g 0.85
```

### Protocol

- MYSTIC backward: 1e7 photons at SZA 95/97 (achieved rel SE 1.1-6.4%),
  1e8 at SZA 99/101 (achieved 3.9-13.4%), radiance and SE from
  mc.rad.spc / mc.rad.std.spc, one run per (tau*, SZA, wavelength),
  each in its own work dir.
- twilight hybrid: 6 seeds x 16000 photons (16k is where the hybrid
  converges at SZA 95 under the deck; at 4000 photons single seeds
  scatter by 2.7x). tau* 1 and 3.
- twilight multiple: 6 seeds, photon tiers 500k (SZA 95/97) and 2e6
  (SZA 99/101) per run. tau* 1, 3, 10 (tau* 10 restricted to SZA 95/97
  for runtime, per the drop-the-deepest-first rule).
- Seed scatter (SE of the 6-seed mean) is the twilight-side error.
- Gate band per point, honestly derived from both sides' SEs:
  `band = 3 x sqrt(se_tw^2 + se_MYSTIC^2) + 5% x MYSTIC` (the 5% floor
  covers solar-table interpolation, the 120 vs 150 km top, and the
  scalar-vs-Stokes residual). Verdicts:
  - Multiple is gated everywhere, except a point is declared LOW-POWER
    (reported, not gated) when its band exceeds 50% of the referee
    value: the comparison has no statistical power there.
  - Hybrid is gated at SZA 95; at SZA >= 97 under a deck it is the
    DOCUMENTED analog-under-cloud starvation limitation (forced
    collision disabled under any cloud channel), so those rows are
    KNOWN-LIM: reported and characterized, not gated.

### Results (radiances in W/m^2/sr/nm, mean +- SE; ratios twilight/MYSTIC)

Gate: 28 pass / 0 fail / 8 low-power / 18 known-lim.

#### tau* = 1

| SZA | wl | twilight hybrid | twilight multiple | MYSTIC backward | hyb/my | mul/my | band | verdict hyb | verdict mul |
|----:|---:|---:|---:|---:|---:|---:|---:|:--|:--|
| 95 | 450 | 7.478e-05 +- 3.4e-06 | 7.583e-05 +- 3.1e-06 | 7.877e-05 +- 8.9e-07 (1e+07) | 0.949 | 0.963 | 17.1% | PASS | PASS |
| 95 | 550 | 6.110e-05 +- 2.6e-06 | 6.423e-05 +- 1.4e-06 | 6.367e-05 +- 7.4e-07 (1e+07) | 0.960 | 1.009 | 12.6% | PASS | PASS |
| 95 | 650 | 4.910e-05 +- 1.7e-06 | 5.037e-05 +- 1.7e-06 | 5.043e-05 +- 6.0e-07 (1e+07) | 0.974 | 0.999 | 15.6% | PASS | PASS |
| 97 | 450 | 6.439e-06 +- 1.9e-06 | 8.411e-06 +- 4.5e-07 | 7.596e-06 +- 3.0e-07 (1e+07) | 0.848 | 1.107 | 26.7% | KNOWN-LIM | PASS |
| 97 | 550 | 6.581e-06 +- 2.4e-06 | 5.361e-06 +- 4.4e-07 | 6.360e-06 +- 2.6e-07 (1e+07) | 1.035 | 0.843 | 29.3% | KNOWN-LIM | PASS |
| 97 | 650 | 4.545e-06 +- 1.7e-06 | 4.698e-06 +- 4.7e-07 | 4.663e-06 +- 2.1e-07 (1e+07) | 0.975 | 1.007 | 38.1% | KNOWN-LIM | PASS |
| 99 | 450 | 1.710e-07 +- 5.0e-08 | 8.624e-07 +- 8.5e-08 | 9.606e-07 +- 3.7e-08 (1e+08) | 0.178 | 0.898 | 33.9% | KNOWN-LIM | PASS |
| 99 | 550 | 1.240e-07 +- 6.0e-09 | 9.202e-07 +- 1.3e-07 | 7.538e-07 +- 3.1e-08 (1e+08) | 0.165 | 1.221 | 58.4% | KNOWN-LIM | LOW-POWER |
| 99 | 650 | 8.506e-08 +- 1.1e-08 | 5.901e-07 +- 6.9e-08 | 5.154e-07 +- 2.3e-08 (1e+08) | 0.165 | 1.145 | 47.7% | KNOWN-LIM | PASS |
| 101 | 450 | 3.049e-08 +- 1.8e-08 | 1.131e-07 +- 2.0e-08 | 1.584e-07 +- 1.6e-08 (1e+08) | 0.193 | 0.714 | 52.4% | KNOWN-LIM | LOW-POWER |
| 101 | 550 | 2.622e-08 +- 1.5e-08 | 1.085e-07 +- 2.8e-08 | 1.655e-07 +- 1.6e-08 (1e+08) | 0.158 | 0.656 | 63.6% | KNOWN-LIM | LOW-POWER |
| 101 | 650 | 1.746e-08 +- 9.7e-09 | 1.065e-07 +- 4.4e-08 | 7.846e-08 +- 9.1e-09 (1e+08) | 0.223 | 1.358 | 175.9% | KNOWN-LIM | LOW-POWER |

#### tau* = 3

| SZA | wl | twilight hybrid | twilight multiple | MYSTIC backward | hyb/my | mul/my | band | verdict hyb | verdict mul |
|----:|---:|---:|---:|---:|---:|---:|---:|:--|:--|
| 95 | 450 | 5.995e-05 +- 9.7e-06 | 6.573e-05 +- 9.4e-07 | 6.498e-05 +- 8.1e-07 (1e+07) | 0.922 | 1.011 | 10.7% | PASS | PASS |
| 95 | 550 | 5.017e-05 +- 5.4e-06 | 5.273e-05 +- 8.8e-07 | 5.361e-05 +- 6.8e-07 (1e+07) | 0.936 | 0.984 | 11.2% | PASS | PASS |
| 95 | 650 | 4.036e-05 +- 3.5e-06 | 4.248e-05 +- 1.1e-06 | 4.220e-05 +- 5.5e-07 (1e+07) | 0.957 | 1.007 | 13.8% | PASS | PASS |
| 97 | 450 | 8.690e-06 +- 4.2e-06 | 7.249e-06 +- 5.1e-07 | 6.967e-06 +- 2.9e-07 (1e+07) | 1.247 | 1.040 | 30.4% | KNOWN-LIM | PASS |
| 97 | 550 | 3.728e-06 +- 1.6e-06 | 6.163e-06 +- 4.7e-07 | 5.358e-06 +- 2.3e-07 (1e+07) | 0.696 | 1.150 | 34.2% | KNOWN-LIM | PASS |
| 97 | 650 | 2.218e-06 +- 4.9e-07 | 3.860e-06 +- 2.6e-07 | 3.384e-06 +- 1.7e-07 (1e+07) | 0.655 | 1.141 | 32.3% | KNOWN-LIM | PASS |
| 99 | 450 | 1.948e-07 +- 1.1e-07 | 8.671e-07 +- 1.1e-07 | 8.769e-07 +- 3.6e-08 (1e+08) | 0.222 | 0.989 | 43.1% | KNOWN-LIM | PASS |
| 99 | 550 | 5.452e-07 +- 3.4e-07 | 7.099e-07 +- 6.6e-08 | 6.533e-07 +- 2.8e-08 (1e+08) | 0.835 | 1.087 | 38.1% | KNOWN-LIM | PASS |
| 99 | 650 | 6.397e-07 +- 4.0e-07 | 5.397e-07 +- 4.7e-08 | 4.585e-07 +- 2.2e-08 (1e+08) | 1.395 | 1.177 | 39.2% | KNOWN-LIM | PASS |
| 101 | 450 | 4.119e-09 +- 2.4e-09 | 1.864e-07 +- 5.9e-08 | 1.579e-07 +- 1.6e-08 (1e+08) | 0.026 | 1.180 | 121.3% | KNOWN-LIM | LOW-POWER |
| 101 | 550 | 3.551e-09 +- 2.1e-09 | 6.064e-08 +- 1.7e-08 | 9.286e-08 +- 1.1e-08 (1e+08) | 0.038 | 0.653 | 70.0% | KNOWN-LIM | LOW-POWER |
| 101 | 650 | 2.367e-09 +- 1.3e-09 | 5.597e-08 +- 2.2e-08 | 6.784e-08 +- 9.1e-09 (1e+08) | 0.035 | 0.825 | 112.0% | KNOWN-LIM | LOW-POWER |

#### tau* = 10 (Multiple only, per the documented hybrid limitation)

| SZA | wl | twilight multiple | MYSTIC backward | mul/my | band | verdict |
|----:|---:|---:|---:|---:|---:|:--|
| 95 | 450 | 2.818e-05 +- 1.2e-06 | 2.879e-05 +- 5.1e-07 (1e+07) | 0.979 | 18.6% | PASS |
| 95 | 550 | 2.128e-05 +- 5.4e-07 | 2.395e-05 +- 4.3e-07 (1e+07) | 0.888 | 13.6% | PASS |
| 95 | 650 | 1.842e-05 +- 7.3e-07 | 1.782e-05 +- 3.4e-07 (1e+07) | 1.034 | 18.5% | PASS |
| 97 | 450 | 3.601e-06 +- 2.4e-07 | 3.320e-06 +- 2.0e-07 (1e+07) | 1.085 | 33.2% | PASS |
| 97 | 550 | 2.647e-06 +- 2.2e-07 | 2.165e-06 +- 1.4e-07 (1e+07) | 1.222 | 41.4% | PASS |
| 97 | 650 | 2.048e-06 +- 2.8e-07 | 1.619e-06 +- 1.2e-07 (1e+07) | 1.265 | 60.7% | LOW-POWER |

### Reading of the results

Multiple (the independent analog estimator): 22 of its 30 points are
gated and ALL pass. Mean gated ratio 1.035, range 0.84-1.22. At SZA 95
it sits within 1-11% of MYSTIC at every tau* including 10 (deep-deck
diffusion, ratio 0.89-1.03). At 97 and 99 it tracks the referee within
the 27-48% bands (the bands are SE-dominated: multiple's own seed
scatter at 2e6 photons is the largest term). There is a mild high-side
trend with tau* at SZA 97 (tau* 1: ~0.98 average, tau* 3: ~1.11,
tau* 10: ~1.19, all within their bands); worth watching if the tau*-10
twilight band is ever tightened with bigger budgets. At SZA 101 all six
points are LOW-POWER (bands 52-176%): the ratios scatter around unity
(0.65-1.36) with no systematic sign, so there is no evidence of
disagreement, but the comparison cannot certify a 10-20% level there
on either side's current budgets.

Hybrid (the production estimator): passes under the deck at SZA 95
(0.92-0.97x at tau* 1 and 3; its bands there are 16-50%, dominated by
the hybrid's own seed scatter, which grows with tau* at twilight
geometry; the 0.92x point sits just inside the low-power threshold and
passes with margin). From SZA 97 the documented analog-under-cloud
starvation is confirmed externally, and the campaign characterizes its
two phases: at SZA 97 the failure mode is VARIANCE BLOW-UP (seed SE
22-48% of the mean; per-point ratios scatter 0.66-1.25 around the
referee with no stable sign at 16k photons), and from SZA 99 it is the
one-sided deficit (tau* 1: 0.16-0.18x at 99, stable across wavelengths
with seed SE down to 5%, and 0.16-0.22x at 101; tau* 3 at 101: collapse
to 0.03-0.04x, values 25-30x below the referee, far beyond seed
scatter). The 0.37-0.45x class seen in the earlier vz-80
arbitration lies between these phases. Nothing here is new machinery
failing: it is the documented limitation, now bounded on a grid. The
combined-channel forced mode for decks remains the tracked Stage-2
follow-up, and these 18 KNOWN-LIM rows are its acceptance targets.

### Runtime notes

Wall clock 2 h 11 min (15:23 to 17:34 CEST) on a 12-logical-core Apple
Silicon machine that was HEAVILY shared for the whole window (system
load average 39-77: two concurrent unrelated validation campaigns).
5 parallel uvspec workers, twilight phases run concurrently with the
referee pool. Unloaded per-case uvspec cost scales as ~6 s per 1e6
photons at tau* 1, ~12 s at tau* 3, ~35 s at tau* 10 (backward + vroom);
the twelve 1e8 deep runs dominate. The runner caches completed MYSTIC
cases (validation/g3/<case>/case.done) so an interrupted campaign
resumes, and a killed deep run degrades to a NO-REF row instead of
aborting the campaign.

### Verdict (Campaign A, for the confidence table)

Clouds at twilight geometry: EXTERNALLY REFEREED in 1D against spherical
backward MYSTIC on the delta-scaled same-problem deck, zenith view,
450/550/650 nm. The analog Multiple estimator is validated to SZA 99 for
tau* up to 3 and to SZA 97 for tau* 10 (22/22 gated points pass, mean
ratio 1.03, bands 11-48% dominated by the two MC noise floors; at
SZA 101 both sides are variance-limited and the six ratios scatter
0.65-1.36 around unity with no systematic sign). The hybrid production
estimator passes under the deck at SZA 95 (0.92-0.97x) and its
documented analog-under-cloud starvation past SZA 97 is now externally
bounded: variance blow-up at 97, one-sided 0.16-0.22x at 99-101
(tau* 1) collapsing to 0.03x (tau* 3, SZA 101); use Multiple
under decks past SZA 97 until the combined-channel forced mode lands.

## Campaign B: g3-cube (transport plan G2b), decks prepared, blocked

Goal was a synthetic 3D cube at daytime SZA vs MYSTIC 3D. Two
independent blockers make the run impossible with public tooling; both
are documented empirically and the complete campaign kit is delivered
ready to run.

### Blocker 1: the public referee cannot do 3D

The public libRadtran 2.0.6 ships MYSTIC with HAVE_MYSTIC3D compiled
out. Probed two ways (runner `--tier g3-cube` reproduces both):

- `mc_sample_grid` is not tokenized in src/uvspec_lex.l at all: any deck
  carrying it fails at parse time ("Unknown command"). Without it, the
  full 16x16 backward deck crashes in setup (SIGBUS) before transport.
- A minimal 2x2 single-pixel 3D deck parses, loads the 3D cloud, and
  aborts on the first photon with "Error! you are not allowed to use
  mystic 3D!" (libsrc_c/mystic.c travel_tau guard, rc 255).

Full 3D MYSTIC is distributed by LMU on collaboration terms only. SHDOM
or the I3RC community MC are the realistic public referee alternatives;
neither is installed here.

### Blocker 2: the twilight CLI has no field-radiance surface

`twilight-cli compare` (the radiance surface every referee tier drives)
does not accept `--cloud-field`; only `pray` does, and pray emits prayer
times, not radiances. External refereeing of the voxel transport path
needs `compare --cloud-field` (or an equivalent radiance-grid surface).
This is the exact CLI gap to close before the cube campaign can run.

### Delivered kit (validation/g3cube/, regenerate with --tier g3-cube)

- `wc3d_cube.dat`: 16x16x1 cells of 1x1x1 km, cloud block cells 7-10 in
  x and y (4x4 km) between 1 and 2 km, per-cell EXPLICIT delta-scaled
  gray optics (MYSTIC ASCII flag 1 = ext, g, ssa: the same-problem
  construction with no microphysics in the loop): ext* = 3/km,
  g* = 0.4350282, ssa* = 0.9293930. These are the ice-preset scaled
  values of the twilight field path (cloud_field_builder.rs: ssa 0.97,
  g 0.77, de-scale 0.424887). The domain is periodic in MYSTIC; the
  block sits 6 km from its periodic images, so daytime shadows at
  SZA <= 60 (shadow throw from 2 km top: 3.5 km) cannot wrap. The
  residual geometry difference vs twilight (spherical shells, finite
  field footprint, non-periodic) is stated here and must accompany any
  future gate on these decks.
- 8 decks `g3cube_sza{30,60}_vz{0,60}_{cloud,clear}.inp`: backward
  MYSTIC, single sample pixel under the cube center (8,8) and in the
  clear half (2,2), zenith and 60-degree slant views, 550 nm, 4e6
  photons, same TSIS solar table, albedo 0.15, Rayleigh-only gas.
- `cube_field.bin(.json)`: the byte-matched twilight Cloud3DField
  sidecar for the SAME cube via tools/cloud3d_common.write_field, IWC
  0.129493 g/m^3 in the block (the inversion of beta = 3 IWC /
  (2 rho_ice r_eff) with r_eff 30 um so the carried scaled extinction is
  exactly ext* = 3/km), domain centered on lat 0, lon 0 so the observer
  for the cloud-center pixel is at (0, 0). VERIFIED loadable by the
  production CLI: `pray --cloud-field validation/g3cube/cube_field.bin`
  reports "3D cloud field: 64 voxels (64x16x16)" (the 4x4 block
  regridded onto 4 internal 250 m levels) and runs the CPU reference
  scan.

### Verdict (Campaign B, for the confidence table)

Clouds in true 3D vs an external 3D referee: NOT YET REFEREEABLE, and
now precisely bounded rather than vaguely aspirational. The public
libRadtran build refuses 3D transport (HAVE_MYSTIC3D compiled out,
demonstrated empirically) and the twilight CLI exposes no field-radiance
surface (compare lacks --cloud-field). The complete same-problem cube
kit (per-cell scaled-optics referee decks + byte-matched Cloud3DField
sidecar, load-verified) is prepared in validation/g3cube/ and runs the
day either a full-MYSTIC/SHDOM referee or the compare surface exists;
until then the 3D voxel path remains anchored by the 1D external
referees above plus the internal field-vs-1D consistency gates.
