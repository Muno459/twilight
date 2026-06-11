export const meta = {
  name: 'twilight-review',
  description: 'Exhaustive physics + code + GPU review of the twilight RT prayer-time engine',
  phases: [
    { title: 'Review', detail: '14 specialist reviewers, one per subsystem, in parallel' },
    { title: 'Verify', detail: 'adversarially refute every critical/high finding' },
    { title: 'Strategy', detail: 'libRadtran validation harness + integration architecture' },
    { title: 'Critic', detail: 'completeness critic over the consolidated findings' },
  ],
}

const ROOT = '/Users/mostafamahdi/twilight'

// Optional model override for the whole sweep (e.g. {model: 'fable'} for a Fable 5 pass).
// Undefined → agents inherit the session model.
const MODEL = (typeof args !== 'undefined' && args && args.model) || undefined
const TAG = MODEL ? `[${MODEL}] ` : ''

const PREAMBLE = `You are a rigorous reviewer of the \`twilight\` Rust project at ${ROOT}.
twilight is a Monte Carlo radiative-transfer (RT) engine that computes Islamic prayer times (Fajr/Isha)
by physically simulating sunlight scattering through a 50-shell spherical atmosphere, then mapping
spectral sky radiance through a CIE vision model to twilight thresholds. It is meant to be used by the
Muslim community ("the ummah"), so CORRECTNESS MATTERS more than cleverness.

The maintainer believes large parts are "AI slop": physically incorrect, fabricated capabilities,
and that the Monte Carlo path has excessive variance, and that the Metal GPU shader does not work.

YOUR JOB: rigorously review ONE assigned dimension. ACTUALLY READ the source files with Read/Grep —
do not guess. Report ONLY findings you can back with a quoted snippet and a precise file:line.
Ground every physics claim in named literature or standard convention (e.g. Bodhaine 1999 Rayleigh,
Bucholtz 1995, HITRAN line shapes, Humlicek 1982 / Weideman 1994 Voigt, DISORT/libRadtran/MYSTIC
conventions, CIE V(λ)/V'(λ) photometry, OPAC aerosols, Mie vs Henyey-Greenstein).

CRITICAL DISCIPLINE: distinguish a STANDARD, ACCEPTABLE APPROXIMATION (normal in RT — do NOT flag)
from an ACTUAL ERROR (wrong sign, wrong units, non-physical magic constant, broken estimator,
fabricated/dead code, a claim contradicted by the code). Be concrete and skeptical. It is better to
report 4 well-evidenced findings than 15 speculative ones.

Severity:
- critical = physics that is wrong in a way that materially shifts prayer times, OR code that cannot
  work at all (won't compile, NaN output, fabricated capability presented as real).
- high     = significant correctness, variance, or fabrication issue.
- medium   = accuracy or quality concern that should be fixed.
- low      = minor.

ESTABLISHED GROUND TRUTH (verified by the orchestrator — build on it, do NOT re-verify):
- Baseline \`cargo build --release --workspace\` (default features) PASSES.
- crates/twilight-gpu/shaders/ contains ONLY twilight.metal. The files twilight.cu, twilight.wgsl,
  and single_scatter.spv / mcrt_trace.spv / hybrid_scatter.spv / garstang_zenith.spv DO NOT EXIST.
  Therefore \`--features cuda\`, \`--features vulkan\`, \`--features webgpu\` FAIL TO COMPILE (confirmed:
  "couldn't read ... No such file or directory"). Only the Metal backend can build.
- GPU tests use \`let Some(gpu) = try_metal() else { return };\` so they SILENTLY PASS as no-ops when
  no GPU device is present, and the vulkan/cuda/webgpu parity tests are behind features that can't compile.`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'summary', 'findings'],
  properties: {
    dimension: { type: 'string' },
    summary: { type: 'string', description: 'Overall health of this subsystem in 2-4 sentences.' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title', 'severity', 'category', 'file', 'lines', 'evidence', 'why_wrong', 'recommendation', 'confidence'],
        properties: {
          id: { type: 'string', description: 'short slug e.g. scotopic-fudge' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category: { type: 'string', enum: ['physics-incorrect', 'fabrication', 'variance', 'bug', 'integration', 'code-quality', 'claim-overstated', 'test-quality'] },
          file: { type: 'string' },
          lines: { type: 'string' },
          evidence: { type: 'string', description: 'Quote the actual offending code/comment.' },
          why_wrong: { type: 'string', description: 'Why it is wrong, with literature/convention reference.' },
          recommendation: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['finding_id', 'verdict', 'reasoning', 'corrected_severity'],
  properties: {
    finding_id: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'partial'] },
    reasoning: { type: 'string', description: 'What you found when you read the code yourself.' },
    corrected_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
    correction: { type: 'string', description: 'If partial/refuted, the accurate statement.' },
  },
}

const DIMENSIONS = [
  {
    key: 'core-physics',
    label: 'single-scatter physics',
    files: 'crates/twilight-core/src/single_scatter.rs, geometry.rs, atmosphere.rs, spectrum.rs, crates/twilight-data/src/atmosphere_profiles.rs',
    focus: `The deterministic single-scatter integrator — the backbone of prayer times.
Verify: (a) Rayleigh cross-section & King correction (spectrum.rs ~lines 5-27 admits an "approximate
F" — check it against Bodhaine 1999 Table). (b) ray-sphere geometry at Earth scale. (c) atmospheric
refraction via Snell's law at shell boundaries — is it ACTUALLY applied in the LOS/shadow integration
or only claimed? does it bend toward the correct direction and conserve? (d) optical-depth integration
& shadow-ray transmittance (single_scatter.rs ~559-565 claims "straight-line approximation is exact to
f64" — is that true given refraction?). (e) the single-scatter source function = solar irradiance ×
transmittance(sun→point) × scattering coeff × phase(scattering angle) × transmittance(point→observer);
verify each factor and units (W/m²/nm/sr). (f) 50-shell non-uniform discretization — adequate near the
tangent point at twilight SZA 90-108° where the path is nearly horizontal? (g) ground/Lambertian albedo.`,
    seeds: `spectrum.rs King factor "Approximate: F ≈ 1.049 at 400nm" (line ~27); single_scatter.rs:559-565
"straight-line approximation is exact to f64 precision".`,
  },
  {
    key: 'mc-variance',
    label: 'Monte Carlo / hybrid + variance',
    files: 'crates/twilight-core/src/photon.rs (7600 lines), path_guide.rs, scattering.rs',
    focus: `THE MAINTAINER'S "LOTS OF VARIANCE" CONCERN. photon.rs is the backward MC tracer with
next-event estimation (NEE) and the hybrid engine (exact order-1 + MC orders 2+). Verify:
(a) Is the estimator UNBIASED? The "zenith bias" (ZENITH_BIAS_N=5.0 ~line 886) and Dwivedi-type
horizontal direction biasing (~line 978) modify sampling — are the photon WEIGHTS corrected by the
ratio pdf_unbiased/pdf_biased so the expectation is preserved, or does this silently bias the result?
(b) NEE correctness (phase × transmittance to sun / pdf). (c) Russian roulette / weight-window —
present? unbiased? (d) WHERE DOES VARIANCE COME FROM: default photon count (~500?), singular weights,
the terminator region, forced_tau_min hacks, missing stratification, RNG quality. (e) Does MC variance
make prayer times non-deterministic / jittery between runs? (f) HG Mueller matrix approx
(scattering.rs ~340-359, P22=P11) — acceptable? (g) path_guide.rs DEFENSIVE_EPSILON (~758). Recommend
concrete variance-reduction (more NEE, control variates, stratified SZA, antithetic, higher photon
default, or replace MC with a deterministic higher-order solver).`,
    seeds: `photon.rs ZENITH_BIAS_N=5.0 (~886), "No bias is introduced" comment (~831), Dwivedi biasing
(~978); recent commits removed "biasing clamps" from Metal — check the CPU side too.`,
  },
  {
    key: 'gas-absorption',
    label: 'molecular gas absorption',
    files: 'crates/twilight-core/src/gas_absorption.rs, crates/twilight-data/src/ozone_xsec.rs, data/xsec/, tools/gen_gas_xsec.py (gas_absorption_data.rs is generated — spot-check only)',
    focus: `Five-species absorption (O3, O2, H2O, NO2, O4 CIA). Verify:
(a) Voigt profile via Humlicek (1982) w4 (~274-359) and Weideman (1994) N=32 Faddeeva (~359) —
correct region boundaries, no discontinuities, correct normalization. (b) Partition-function ratio
approximated as (T_ref/T)^1.5 (~449-461) — that exponent is the classical diatomic rotational
approximation; is it applied to species where it's invalid (H2O is a nonlinear triatomic → ^1.5 is
wrong, should be ~^1.5 only for linear/diatomic; nonlinear → ^2 roughly)? (c) O3 Serdyuchenko 11-T
interpolation with temp clamp 220-294K (~170) — extrapolation handling. (d) negative cross-section
clamping (~986) — physical or masking a bug? (e) UNITS: cm² vs m², cm⁻¹ line positions vs nm, number
density m⁻³ vs cm⁻³ — a units slip here is catastrophic. (f) Is line-by-line actually integrated over
the 41-wavelength grid correctly, or sampled at band centers? (g) Does gen_gas_xsec.py match what the
Rust expects?`,
    seeds: `gas_absorption.rs:170 temp clamp; :449-461 (T_ref/T)^1.5; :986 negative xsec clamp; :317
"Let me use the known 2-term rational approximation" (LLM-ish comment).`,
  },
  {
    key: 'aerosols-clouds',
    label: 'aerosols + clouds',
    files: 'crates/twilight-data/src/aerosol.rs, cloud.rs, builder.rs, solar_spectrum.rs; crates/twilight-core/src/scattering.rs (HG); crates/twilight-clouds (if it exists)',
    focus: `USER PRIORITY: "we need clouds". Verify:
(a) OPAC aerosol optical properties (6 types) — extinction, SSA, asymmetry g, Angstrom exponent —
plausible vs OPAC/Hess 1998? (b) Aerosol phase function = single-term Henyey-Greenstein — crude but
standard; acceptable. (c) CLOUDS: are clouds modeled with HG too? Real cloud droplets need a Mie phase
function (sharp forward peak, glory/fogbow backscatter, g~0.85) — a single HG badly misrepresents cloud
multiple scattering and the diffuse twilight glow under cloud. Flag if clouds use HG. (d) Are cloud
optical depths, droplet effective radii, base/top altitudes physical for the 6 named types
(thin-cirrus..cumulus)? (e) INTEGRATION: is cloud actually baked into the atmosphere shells and fed to
BOTH the CPU and Metal RT paths, or is it CPU-only / decorative? Does --cloud measurably and correctly
shift prayer times? (f) Is there a twilight-clouds crate at all, or is it vapor?`,
    seeds: `crates/twilight-clouds was NOT in the line-count inventory — verify whether it exists/empty.
README claims "Cloud layers (6 types)" and clouds in the GPU path.`,
  },
  {
    key: 'vision-threshold',
    label: 'vision + prayer thresholds',
    files: 'crates/twilight-threshold/src/threshold.rs, luminance.rs, vision.rs',
    focus: `Maps spectral radiance → perceived luminance → Fajr/Isha. Verify:
(a) luminance_scotopic = luminance * 1.5 (threshold.rs:487 and :615) — scotopic luminance is NOT
photopic × 1.5; it requires re-integrating spectral radiance against V'(λ) (scotopic luminous
efficiency, peak 507nm) with the scotopic luminous-efficacy constant (1700 lm/W), totally different
from photopic V(λ) (683 lm/W, peak 555nm). A flat ×1.5 is a fabricated magic factor — quantify the
error. (b) Mesopic model (CIE 191:2010 / MOVE) — is it real or invented? (c) Fajr threshold and Isha
al-abyad / al-ahmar thresholds — are the luminance/contrast values grounded in any published twilight
photometry or fiqh-referenced observation, or are they arbitrary tuned constants? (d) shafaq color
classification via spectral centroid — defensible boundary? (e) CIE LUT correctness (vision.rs V(550)
vs V(560) sanity).`,
    seeds: `threshold.rs:487 & :615 luminance_scotopic = luminance * 1.5; luminance.rs:601 "small
tolerance" l_red+l_blue < l_total*1.01.`,
  },
  {
    key: 'metal-gpu',
    label: 'Metal shader correctness / why it fails',
    files: 'crates/twilight-gpu/shaders/twilight.metal, crates/twilight-gpu/src/metal.rs, crates/twilight-cpu/src/gpu_dispatch.rs, crates/twilight-gpu/src/tests.rs, oracle.rs',
    focus: `MAINTAINER SAYS "the metal shader doesn't work." Investigate WHY. There are UNCOMMITTED
changes — run \`cd ${ROOT} && git diff --stat\` and \`git diff crates/twilight-gpu/shaders/twilight.metal\`
and \`git log --oneline -12\` to see the active debugging thread (a "v2 ray-parallel" kernel produced
NaNs, fell back to v1; "biasing clamps" were removed; 5000-ray split dispatch + GPU timeout detection
were added). Verify: (a) Does GPU single-scatter match the CPU f64 oracle within stated tolerance, or
does it diverge? (b) The hybrid/MCRT Metal path — does it produce usable radiance or NaN/zeros/noise?
(c) splitmix64 RNG on GPU — seeded per-ray correctly, decorrelated? (d) Were the removed "biasing
clamps" masking a real NaN source (div-by-zero, log(0), negative sqrt, exp overflow)? (e) Does the
prayer pipeline actually dispatch to Metal, or silently fall back to CPU? (f) Is the f32 precision
story (Kahan, half-b, boundary snapping) real and correct, or partially decorative? Cite exact
.metal line numbers.`,
    seeds: `Commits: "v2 NaN under investigation", "Remove all biasing clamps from Metal shader",
"Fix 5000-ray Metal prayer: split dispatch + GPU timeout detection". gpu_dispatch.rs and tests.rs are
in the uncommitted diff.`,
  },
  {
    key: 'gpu-fabrication',
    label: 'GPU backend fabrication',
    files: 'crates/twilight-gpu/src/vulkan.rs, cuda.rs, wgpu_backend.rs, lib.rs, crates/twilight-gpu/Cargo.toml, README.md (GPU section)',
    focus: `GROUND TRUTH (given): only twilight.metal exists; cuda/vulkan/webgpu features CANNOT COMPILE
(missing .cu/.wgsl/.spv). Your job: ENUMERATE every fabricated/overstated claim and assess the dead
code. Specifically: (a) README line ~131 "Four hand-tuned backends with native shaders written from
scratch for each API" and ~140 "Each backend has hand-written shaders"; (b) the entire GPU feature
TABLE (README ~142-155) — every Vulkan/CUDA/WebGPU column cell asserts features that don't exist;
(c) ~131 "Cross-backend parity tests with CPU f64 oracle ensure all backends produce identical physics"
— impossible, they don't compile; (d) roadmap "[x] GPU backends (Metal, Vulkan, CUDA, wgpu) with
cross-backend parity"; (e) the GPU badge. Then assess: are vulkan.rs/cuda.rs/wgpu_backend.rs coherent
host implementations that merely lack shaders, or are they ALSO stubbed/hallucinated (check that the
buffer layouts, dispatch logic, trait impls are real)? Recommend the honest path (delete the vapor
backends + fix README, or actually author the shaders).`,
    seeds: `vulkan.rs include_bytes! 4 missing .spv; cuda.rs include_str!("../shaders/twilight.cu");
wgpu_backend.rs include_str!("../shaders/twilight.wgsl"); wgpu_backend.rs comment "reserved for Phase
11f pipeline integration".`,
  },
  {
    key: 'solar',
    label: 'solar position (SPA + DE440)',
    files: 'crates/twilight-solar/src/spa.rs, de440.rs, spk.rs, earth_rotation.rs, spa_tables.rs',
    focus: `Likely the most solid subsystem — confirm or find overstatement. Verify: (a) SPA (NREL
Reda & Andreas 2004) implementation — VSOP87 terms, nutation/aberration, topocentric correction;
is there a test against the canonical NREL reference case (2003-10-17 Z, lat 39.742476...)? (b) DE440
DAF/SPK reader + Chebyshev interpolation + IAU precession-nutation — claims "validated to 8 m vs JPL
Horizons" and "±0.001 arcsec"; are these backed by an actual test with reference values, or asserted in
prose? (c) refraction model for sunrise/sunset and the binary search; high-latitude persistent-twilight
handling. Flag claims that lack a verifying test.`,
    seeds: `README "DE440 validated to 8 meters vs JPL Horizons", "±0.0003 degrees", "sub-arcsecond".`,
  },
  {
    key: 'skyglow',
    label: 'light pollution / skyglow',
    files: 'crates/twilight-skyglow/src/garstang.rs, spectrum.rs, bortle.rs, angular.rs, lib.rs',
    focus: `Garstang RT skyglow added to the twilight signal. Verify: (a) Garstang (1986/1989/1991)
model implementation — aerosol+molecular scattering of city light, double-scattering, the
characteristic integrals; correct vs the published formulation? (b) Bortle class (1-9) → zenith
radiance/luminance mapping (bortle.rs) — a real calibrated mapping (e.g. to mag/arcsec² or mcd/m²) or
invented constants? (c) VIIRS radiance mapping. (d) LED/HPS spectral profiles. (e) UNITS and geometry:
does the skyglow zenith brightness get added to the sky luminance in consistent units before the
threshold test, and does that addition actually shift Fajr/Isha, or is it decorative? `,
    seeds: `README "Garstang RT skyglow model", "Estimates zenith brightness added to the twilight
signal and the resulting shift in prayer times".`,
  },
  {
    key: 'terrain',
    label: 'terrain masking',
    files: 'crates/twilight-terrain/src/projection.rs, horizon.rs, geotiff.rs, copernicus.rs, lidar/denmark.rs, cache.rs, lib.rs',
    focus: `Verify: (a) GeoTIFF parser correctness (tags, tiling, DEM elevation decode). (b) 360-point
horizon-profile computation — geometry of horizon elevation angle from DEM around the observer,
including Earth curvature & (optionally) refraction over distance. (c) The "effective solar zenith
angle adjustment per azimuth": is the sun's azimuth at twilight matched to the horizon profile correctly
so that a mountain in the EAST delays Fajr / a mountain in the WEST advances Isha (signs!)? A sign or
azimuth-convention error here would shift times wrongly. (d) Copernicus GLO-30 tile download/caching;
Danish SDFI LiDAR. (e) Does terrain integrate into the prayer pipeline or only the CLI demo?`,
    seeds: `README "Computes a 360-point horizon profile and adjusts the effective solar zenith angle at
each azimuth."`,
  },
  {
    key: 'weather',
    label: 'live weather mapping',
    files: 'crates/twilight-weather/src/mapping.rs, lib.rs, api.rs',
    focus: `Open-Meteo → atmosphere overrides. Verify: (a) AOD@550nm → aerosol optical properties: does
it pick an OPAC type and scale, and is the Angstrom extrapolation to other wavelengths correct? (b)
Cloud cover % by altitude band → cloud layers: is mapping % to optical depth physical or arbitrary?
(c) Surface O3 (µg/m³) → column Dobson Units and surface NO2 (µg/m³) → number density (m⁻³): these unit
conversions (mapping.rs, ~665 lines) are error-prone — verify the molar masses, Avogadro, the DU
definition (2.687e20 molecules/m² per DU), and whether a SURFACE concentration can validly be turned
into a COLUMN (it cannot without a profile assumption — flag the assumption). (d) Are CAMS fields fetched
with correct Open-Meteo variable names/units?`,
    seeds: `README "maps measured AOD at 550nm ... surface O3/NO2 concentrations from CAMS to gas
absorption overrides (O3 column in Dobson Units, NO2 number density)".`,
  },
  {
    key: 'cpu-pipeline',
    label: 'CPU pipeline + threshold search + integration',
    files: 'crates/twilight-cpu/src/pipeline.rs, simulation.rs, tracer.rs, gpu_dispatch.rs',
    focus: `THE "PROPER INTEGRATION" CONCERN. Trace the end-to-end prayer pipeline. Verify: (a) two-pass
adaptive search (coarse 0.5° then fine 0.1° around crossings) — correct and convergent? (b) Does the
radiance at each SZA come from GPU or CPU, single-scatter or hybrid? If hybrid MC, the threshold
crossing is being found on a NOISY function — does MC variance produce a jittery / non-reproducible
crossing SZA (→ unstable prayer minute)? Quantify. (c) binary search SZA→clock time via SPA — monotonic,
correct branch (morning vs evening)? (d) Are aerosols/clouds/refraction/polarization/terrain/skyglow all
actually threaded into this pipeline, or are some only reachable from separate CLI subcommands and never
combined? (e) Adaptive photon/SZA-step logic.`,
    seeds: `pipeline.rs:506,622 approximate crossing regions; gpu_dispatch.rs is in the uncommitted diff.`,
  },
  {
    key: 'test-quality',
    label: 'test suite honesty (978 claim)',
    files: 'all crates *tests* + #[test] fns across workspace; README Tests section',
    focus: `Is "978 tests passing" honest? Verify: (a) Sample tests across crates — how many assert real
physics/values vs trivial ("doesn't panic", "is_finite", ">= 0")? (b) Count GPU tests that early-return
\`else { return }\` when no device — these PASS as no-ops; on CI with no GPU the whole twilight-gpu suite
is silent no-ops. (c) The "cross-backend parity" and vulkan/cuda/webgpu tests are behind features that
DON'T COMPILE — so they never run; is the 978 count even reachable in one build? (d) Are there
golden/reference-value tests (vs SPA NREL case, vs HITRAN xsec, vs a known radiance) or mostly
self-consistency? (e) Does the README test-count-per-crate table match reality (run \`cargo test\` per
crate to count, or grep #[test])? Estimate the fraction of "load-bearing" tests.`,
    seeds: `tests.rs uses \`let Some(gpu)=try_metal() else { return };\` pattern; README claims 311 core,
153 data, 139 gpu, etc., 978 total.`,
  },
  {
    key: 'slop-sweep',
    label: 'AI-slop / fabrication sweep (cross-cutting)',
    files: 'whole repo — grep for dead code, false comments, fabricated numbers, cruft',
    focus: `Cross-cutting "AI slop" audit. Find: (a) dead code (#[allow(dead_code)], fields "reserved for
Phase 11f", unused backends/structs). (b) comments that LIE about or contradict the code, or read like
chat transcripts ("Let me use...", "Our HG approximation..."). (c) FABRICATED benchmark/perf numbers —
is there ANY benchmark harness producing the README's "30 ms", "8.4 ms GPU", "8.9 s", "843% utilization",
"2.5x faster", or are they invented? (d) the "978 tests" / "tests passing" badge accuracy. (e) v1/v2
kernel cruft, duplicated logic, copy-paste. (f) claims of features (polarization, DE440, terrain) that
exist as code but are never wired into \`pray\`. Give an overall "slop ratio" judgment and name the 5
worst offenders. Use grep widely.`,
    seeds: `README badge "tests-978_passing"; perf table M2 Pro numbers; wgpu_backend.rs "reserved for
Phase 11f"; photon.rs/gas_absorption.rs transcript-style comments.`,
  },
]

function finderPrompt(d) {
  return `${PREAMBLE}

=== YOUR DIMENSION: ${d.label} (${d.key}) ===
Primary files: ${d.files}

What to scrutinize:
${d.focus}

Seed observations from the orchestrator (dig deeper, confirm or refute, find more — these are leads,
not conclusions):
${d.seeds}

Read the files. Return a structured review of THIS dimension only. Every finding MUST quote real code
and cite file:line. Set confidence honestly (low if you couldn't fully verify). Order findings by
severity. If the subsystem is actually sound, say so and report few/no findings — do not invent issues.`
}

function verifierPrompt(d, f) {
  return `${PREAMBLE}

You are an ADVERSARIAL VERIFIER. A reviewer of the "${d.label}" subsystem reported this finding:

  id: ${f.id}
  title: ${f.title}
  severity: ${f.severity}  category: ${f.category}
  file: ${f.file}  lines: ${f.lines}
  evidence: ${f.evidence}
  why_wrong: ${f.why_wrong}

Your job: try to REFUTE it. Open ${f.file} yourself, read the cited lines AND surrounding context,
and decide if the problem is real. A finding is "confirmed" ONLY if (1) the code truly does what the
finding says AND (2) the physics/logic reasoning holds against standard literature/convention. Mark
"refuted" if the code doesn't actually do that, or it's a standard acceptable approximation, or the
reviewer misread it. Mark "partial" if the core issue is real but the severity/explanation is off.
Default to skepticism. Give the corrected severity. Cite what you actually saw.`
}

// ---- Strategy agent prompts ----
const LIBRADTRAN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['approach', 'comparison_quantities', 'libradtran_setup', 'twilight_changes_needed', 'tolerances', 'blockers', 'steps'],
  properties: {
    approach: { type: 'string' },
    comparison_quantities: { type: 'array', items: { type: 'string' } },
    libradtran_setup: { type: 'string', description: 'A concrete uvspec input deck sketch for matched twilight geometry.' },
    twilight_changes_needed: { type: 'string', description: 'What CLI/output hooks twilight needs to emit comparable spectral radiance/irradiance.' },
    tolerances: { type: 'string' },
    blockers: { type: 'array', items: { type: 'string' } },
    steps: { type: 'array', items: { type: 'string' } },
  },
}

const INTEGRATION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['current_dataflow', 'gaps', 'variance_assessment', 'recommended_architecture', 'priority_roadmap'],
  properties: {
    current_dataflow: { type: 'string', description: 'Traced end-to-end pipeline as it exists.' },
    gaps: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['area', 'problem', 'impact', 'fix'],
        properties: { area: { type: 'string' }, problem: { type: 'string' }, impact: { type: 'string' }, fix: { type: 'string' } },
      },
    },
    variance_assessment: { type: 'string' },
    recommended_architecture: { type: 'string' },
    priority_roadmap: { type: 'array', items: { type: 'string' } },
  },
}

const libradtranPrompt = `${PREAMBLE}

You are the VALIDATION-STRATEGY agent. The maintainer wants twilight compared against a trusted
reference RT model — libRadtran (uvspec), the community standard. Design a CONCRETE, runnable
validation plan so twilight's physics can be checked, not just asserted.

Read enough of crates/twilight-core (single_scatter.rs, atmosphere.rs, spectrum.rs) and twilight-cli
to know exactly what twilight can output and in what units/geometry. Then design:
- The observables to compare (e.g. spectral sky radiance toward zenith and along the solar principal
  plane at SZA 90-108°; direct+diffuse spectral irradiance; integrated luminance) and WHY each tests a
  different part (Rayleigh, gas absorption, aerosol/cloud multiple scattering).
- A matched libRadtran uvspec input deck (atmosphere_file US Std 1976, mol_modify O3 to the same DU,
  aerosol_default / aerosol_species, sza, wavelength grid 380-780nm, rte_solver disort AND mystic for
  the MC apples-to-apples, umu/phi for radiance, output_user). Give a real sketch.
- Exactly what twilight needs to emit to be comparable (a CLI mode dumping per-wavelength radiance at
  specified (umu,phi) and SZA; matching units W/m²/nm/sr; same solar spectrum — note libRadtran's
  default solar file vs twilight's TSIS-1).
- Tolerances (single-scatter vs DISORT single-scatter should match tightly; full field looser).
- Blockers (installing libRadtran, solar-spectrum mismatch, refraction handling, polarization).
- A numbered step list to stand up tools/validate_libradtran.py + the twilight CLI hook.

Be specific enough that an engineer could execute it this week.`

const integrationPrompt = `${PREAMBLE}

You are the INTEGRATION-ARCHITECTURE agent. The maintainer says it "needs proper integration." Trace
the ACTUAL end-to-end data flow of the \`pray\` command by reading crates/twilight-cli/src/main.rs and
crates/twilight-cpu/src/pipeline.rs and how they call twilight-core / twilight-gpu / twilight-data /
twilight-threshold / twilight-terrain / twilight-skyglow / twilight-weather.

Determine, with file:line evidence:
- The real path: solar position -> atmosphere build (gas/aerosol/cloud baked into shells) -> RT (CPU or
  GPU? single-scatter or hybrid?) -> spectral radiance -> vision/luminance -> threshold crossing -> SZA
  -> clock time -> terrain/skyglow adjustments.
- Which advertised features are ACTUALLY combined in \`pray\` vs only reachable from separate subcommands
  or never wired (clouds? polarization? terrain? skyglow? weather? GPU hybrid? DE440?).
- Whether MC variance feeds a noisy threshold search (non-deterministic prayer minute).
- Where the seams are broken or duplicated.
Then propose a clean target architecture and a PRIORITY ROADMAP (ordered) to make it "proper" for
production use by the ummah — correctness first.`

const CRITIC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['missing_or_underexplored', 'highest_risk_unverified', 'overall_assessment'],
  properties: {
    missing_or_underexplored: { type: 'array', items: { type: 'string' } },
    highest_risk_unverified: { type: 'array', items: { type: 'string' } },
    overall_assessment: { type: 'string' },
  },
}

// ================= RUN =================
phase('Review')
log(`Reviewing ${DIMENSIONS.length} subsystems in parallel; verifying high-severity findings; plus libRadtran + integration strategy.`)

const reviewPromise = pipeline(
  DIMENSIONS,
  (d) => agent(finderPrompt(d), { label: `${TAG}review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA, model: MODEL }),
  (result, d) => {
    if (!result || !Array.isArray(result.findings)) return result
    const toVerify = result.findings.filter((f) => f.severity === 'critical' || f.severity === 'high')
    if (toVerify.length === 0) return { dimension: result.dimension, summary: result.summary, allFindings: result.findings, verified: [] }
    return parallel(
      toVerify.map((f) => () =>
        agent(verifierPrompt(d, f), { label: `${TAG}verify:${d.key}:${f.id}`, phase: 'Verify', schema: VERDICT_SCHEMA, model: MODEL })
          .then((v) => ({ finding: f, verdict: v }))
      )
    ).then((verds) => ({
      dimension: result.dimension,
      summary: result.summary,
      allFindings: result.findings,
      verified: verds.filter(Boolean),
    }))
  }
)

const strategyPromise = parallel([
  () => agent(libradtranPrompt, { label: `${TAG}strategy:libradtran`, phase: 'Strategy', schema: LIBRADTRAN_SCHEMA, model: MODEL }),
  () => agent(integrationPrompt, { label: `${TAG}strategy:integration`, phase: 'Strategy', schema: INTEGRATION_SCHEMA, model: MODEL }),
])

const [reviewed, strategy] = await Promise.all([reviewPromise, strategyPromise])
const [libradtran, integration] = strategy || [null, null]

// Build a compact digest of confirmed/high findings for the critic.
phase('Critic')
const digestLines = []
for (const r of (reviewed || []).filter(Boolean)) {
  const conf = (r.verified || []).filter((v) => v.verdict && v.verdict.verdict !== 'refuted')
  const refuted = (r.verified || []).filter((v) => v.verdict && v.verdict.verdict === 'refuted').length
  digestLines.push(`## ${r.dimension}: ${(r.allFindings || []).length} findings, ${conf.length} confirmed high/crit, ${refuted} refuted`)
  for (const v of conf) {
    digestLines.push(`- [${v.verdict.corrected_severity}] ${v.finding.title} (${v.finding.file}) — ${v.verdict.verdict}`)
  }
}
const digest = digestLines.join('\n')

const criticPrompt = `${PREAMBLE}

You are the COMPLETENESS CRITIC. Below is the consolidated digest of confirmed high/critical findings
from a 14-dimension review of twilight. Identify what is MISSING or under-explored: a subsystem or
failure mode not covered, a physical-correctness claim still unverified, a cross-cutting issue (units,
determinism, the gap between README claims and reality) that no single reviewer owned. Be specific and
actionable. Do not repeat the findings — find the GAPS.

DIGEST:
${digest}`

const critic = await agent(criticPrompt, { label: `${TAG}critic`, phase: 'Critic', schema: CRITIC_SCHEMA, model: MODEL })

return { reviewed: (reviewed || []).filter(Boolean), libradtran, integration, critic }
