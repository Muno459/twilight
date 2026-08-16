# Archival and citation for submission

Journals increasingly require that code and data underlying a manuscript
be deposited somewhere immutable and citable; a GitHub URL is not, because
the tree behind it can change or disappear. Both manuscripts currently
point at the repository. This is the checklist for turning that into a
DOI, and the two steps that need the author's own accounts.

## What is already in place

- `CITATION.cff` at the repository root, so GitHub renders a "Cite this
  repository" button and Zenodo picks the metadata up automatically.
- MIT OR Apache-2.0 licensing, which satisfies every open-data policy the
  target journals apply.
- Every validation table names the command that regenerates it, and the
  referee caches are committed, so a reader can re-derive the tables
  without rebuilding libRadtran (`validation/README.md`).

## Step 1: link Zenodo to the repository (author, once)

Requires the author's GitHub and Zenodo accounts; it cannot be scripted
from outside them.

1. Sign in at <https://zenodo.org> with GitHub.
2. Under Account -> GitHub, flip the switch on `Muno459/twilight`.
3. Zenodo now watches for GitHub *releases*. Nothing is archived until a
   release is published, so the switch alone does nothing.

## Step 2: publish the release that the papers cite

Tag the exact tree the manuscripts describe. Do this AFTER the deep-tier
numbers in the papers are final, because the DOI must point at the tree
that produced them.

```bash
git tag -a v1.0.0 -m "twilight v1.0.0: the tree behind the methods and application papers"
git push origin v1.0.0
```

Then publish a GitHub release from that tag. Zenodo mints two DOIs:

- a **concept DOI** that always resolves to the newest version, and
- a **version DOI** pinned to v1.0.0.

**Cite the version DOI in the manuscripts.** The concept DOI is the right
thing to put in the README, because it follows the project forward; the
papers must point at the frozen tree.

## Step 3: fold the DOI back in

Once minted, three places need it:

1. `CITATION.cff`: add `doi: 10.5281/zenodo.XXXXXXX` and set
   `version`/`date-released` to the release.
2. Both manuscripts' data-availability statements (`declarations.md`),
   replacing "the public repository referenced in the manuscripts" with
   the DOI.
3. `README.md`, with a Zenodo badge next to the existing ones.

## What to archive beyond the code

The repository already carries the referee caches and the campaign
artifacts, so a release archives them too. Two things are deliberately
NOT in the repository and need a decision before submission:

- `data/de440.bsp` (~114 MB JPL ephemeris) is gitignored. It is
  third-party data with a stable public source, so cite the source rather
  than mirroring it; the README gives the download.
- The per-seed deep-tier JSONs under
  `artifacts/2026-07_deep_closure/deep_tier/` ARE committed, and they are
  what a referee would need to re-derive the deep table. Keep them in.

## Venue note for the application paper

Paper 1 (methods) targets JQSRT, which is a natural fit: a transport
estimator, a variance-reduction stack, and an external validation
program.

Paper 2 (the Quranic dawn/dusk application) is a different kind of
contribution and a different readership. Submitting it as a companion to
JQSRT risks it being judged as a methods paper, which it is not, and
risks paper 1 being judged by association. Venues whose scope actually
matches what paper 2 does - a psychophysical detection criterion applied
to observational campaign data - include applied optics and applied
meteorology titles, and journals of the astronomy-and-society or
science-and-religion type where the observational-campaign comparison is
the contribution. The papers cross-reference each other, which works
whether or not they land in the same place.

This is a judgement call for the author; it is recorded here so it is
made deliberately rather than by default.
