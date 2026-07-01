# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Version control

This repo is a git project used to **track code changes**. Large data trees
(`NIfTIs/`, `NIfTIs_leica/`, `data/`, and all `*.nii`/`*.tif`/`*.rar`) are
`.gitignore`d — do not commit them.

**Before committing, ask the user for permission.** Whenever code (or any tracked
file) has changed, explicitly confirm with the user before staging/committing —
summarize what changed and ask whether to commit. Do not commit or push
automatically, even after finishing a task; wait for the user's go-ahead.

**Generated images go in `miscelaneous/`.** Write any generated images — QC
overlays, MIPs, summary figures, diagnostic plots — into the `miscelaneous/`
folder at the repo root. It is `.gitignore`d (regenerable, not source), so nothing
there is committed.

## What this is

`sc3d` is a 3D per-object metrics pipeline. It walks a tree of segmented microscopy
z-stacks stored as NIfTI, runs connected-component labeling on each segmentation,
and emits one row of shape + fluorescence-intensity + derived metrics per object.
Output is CSV: one `cell_metrics_per_object.csv` next to each stack and one
aggregated `all_samples_metrics.csv` at the root.

This repo is the **downstream metrics stage only**. The upstream stage that
produced the NIfTIs (slice integration / cell counting — note the Windows
`stack_path`s in `NIfTIs/missing_slices_report_*.json`) is not in this repo.

## Commands

The package is **not installed**; everything runs against the source tree via
`PYTHONPATH=src` (or `pip install -e .` to get the `sc3d` console script).

```bash
# Environment (conda)
conda env create -f environment.yml && conda activate sc3d

# Run the full pipeline over a data root (writes per-stack + aggregate CSVs)
PYTHONPATH=src python -m sc3d NIfTIs                 # or: sc3d NIfTIs  (if pip-installed)
PYTHONPATH=src python -m sc3d NIfTIs --aggregate-dir out -v

# Tests (synthetic volumes with analytic ground truth; skips if SimpleITK absent)
PYTHONPATH=src pytest tests/ -q
PYTHONPATH=src pytest tests/test_synthetic.py::test_sphere_metrics_close_to_analytic   # single test

# Per-stack summary figure (reads the per-stack CSV, or --recompute from NIfTI)
python visualize_stack.py "NIfTIs/Day_1/Alginate/well 1_Alginate_z stack 1"
python visualize_stack.py <stack_dir> --recompute --show
```

## Architecture

Flow (in `core/pipeline.py:run`): `loader.discover(root)` → for each stack
`loader.load` → `core/labeling.label_and_filter` → `compute_all` (runs every
registered metric provider) → `output/writer` CSVs.

Three extension seams, each isolated behind a `Protocol`:

1. **Input format — `io/base.StackLoader`.** `discover(root) -> Iterator[StackRef]`
   and `load(ref) -> Stack`. `io/nifti.NiftiStackLoader` is the only impl;
   `io/discovery.walk_segmentations` does the actual tree walk. To support a new
   on-disk format, write a new loader — nothing downstream changes (see
   `test_pluggable_loader_inmemory` for an in-memory loader plugged in unchanged).

2. **Metrics — `metrics/base.MetricProvider`.** Each provider declares `name`,
   `columns`, and `compute(stack, prior) -> {label: {column: value}}`. Providers
   **self-register** with `@register(MetricGroup.X)` and are pulled in by
   importing `sc3d.metrics` (its `__init__` imports `shape`, `intensity`,
   `derived`). To add a metric, write a provider, decorate it, and add its columns
   to `output/schema.CANONICAL_COLUMNS`. No central wiring to edit.

3. **Group ordering is load-bearing.** `metrics/registry.GROUP_ORDER` is
   `(SHAPE, INTENSITY, DERIVED)` and providers run in that order. `compute_all`
   passes the running merged row map as `prior` to each provider, so `DerivedMetrics`
   reads `shape`/`intensity` outputs (e.g. `centroid_displacement` from geometric
   vs. fluor-weighted centroids). A provider that depends on another's columns must
   be in a later group.

### Key conventions and invariants

- **Labeling semantics (`core/labeling.py`):** any non-zero voxel is foreground;
  components are relabeled **sorted by descending size, so `label 1 = largest`**,
  and components with `< MIN_VOXELS` (10, `config.py`) are dropped as noise. The
  returned `{label: n_voxels}` map defines which labels exist downstream.
- **Canonical column order (`output/schema.py`)** mirrors the existing
  `NIfTIs/all_samples_metrics.csv` header exactly — downstream consumers depend on
  it. The writer reorders to canonical and back-fills missing columns with NaN, so
  column order is decoupled from provider execution order. Keep schema and provider
  `columns` in sync when changing either.
- **`sample` identity** is the stack folder path relative to root, forward-slashed
  (e.g. `Day_1/Alginate/well 1_Alginate_z stack 1`) — stable across OSes.
- **Fluor/segmentation must share physical space.** `NiftiStackLoader.load`
  resamples the fluor image onto the segmentation grid with **nearest-neighbor**
  (preserves raw integer ADU intensities) when size/origin/spacing/direction drift.
  Intensity and `centroid_displacement` metrics are only meaningful because of this.
- **Missing fluor is tolerated:** `IntensityMetrics` emits NaN columns and the
  derived metrics propagate NaN; a stack with no `*_fluor.nii` still produces shape
  rows.
- **Per-stack failures are non-fatal:** `run` logs and skips a stack that raises,
  then continues and still writes the aggregate.
- **`median` is computed manually** in `IntensityMetrics` from the masked numpy
  view — SimpleITK's `LabelIntensityStatisticsImageFilter` does not expose it.

### Data layout (input)

`NIfTIs/Day_{1,3,5}/{Alginate,PA,PA-RC}/<stack folder>/` with, per stack:
`<stem>_segmentation.nii` (required, drives discovery), `<stem>_fluor.nii`
(green channel, optional), `<stem>_raw.nii`. Discovery keys off the
`_segmentation.nii` suffix; the per-stack CSV is written into that same folder.

## Reference

`3D_metrics_description.md` is the **spec** (code comments that say "Spec §N" point
here): exact formulas, units, ranges, and biological interpretation for every
column, plus caveats (header-unit dependence, `fluor_sum` needs flat-field
correction, sphericity sensitivity to segmentation roughness). Consult it before
changing any metric definition.
