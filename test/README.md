# Test suite

Most of the suite runs on CPU with no GPU, no OpenGL driver and no network access. Tests that
need something unavailable — a GL context, a CUDA device, an optional dependency, cached
model weights — skip rather than fail, so a minimal install still gives a green run.

```bash
pip install -r requirements.txt
pip install pytest
pytest -q                       # everything available here
pytest test/backends -v         # backend registry and cross-engine equivalence
pytest test/raster -v           # the torch rasteriser proofs
pytest -m "not slow" -q
```

Coverage depends on what is installed. Measured on this project: **433 passed / 5 skipped**
on Python 3.9 with the ModernGL, torch and embedding extras, and **428 passed / 32 skipped**
on Python 3.11 with all three render backends but no embedding extras.

## Layout

| Directory | What it proves | Needs |
|---|---|---|
| `unit/` | Pure functions: geometry, cameras, coordinate conversion, labels, image helpers. No rendering. | — |
| `raster/` | The torch rasteriser against **independently derived** answers. This is where the tensor replacement for a GL driver is actually proved. | — |
| `backends/` | The backend registry, engine selection, cross-engine equivalence, coverage semantics, N-channel fields, and the ray casters. | — |
| `golden/` | Whole-image fixtures captured from the ModernGL renderer. Every backend must reproduce them. | a GL context |
| `integration/` | `ProjectionScene` end to end on a generated synthetic flight. | — |
| `embedding/` | The field reducers, and DINOv3 extraction when the weights are cached. | `AlfsPy[embedding]` |
| `parity/` | Comparison against a separate 2.x ModernGL checkout. Largely superseded by `backends/`. | `ALFS_REFERENCE_PATH` |
| `bench/` | Throughput measurements. Not collected by pytest. | — |
| `tools/` | `probe_pyrr` — regenerates the matrix-convention table in `docs/`. | — |
| `helpers/` | Synthetic scenes, the flight-dataset generator, tolerance-based image comparison. | — |

## What makes these tests worth trusting

Expected values are **derived, not recorded**. `test/raster/test_rasterizer.py` builds its
coverage masks with plain numpy from the camera matrices; `test_renderer.py` recomputes the
projective UV mapping from pinhole geometry; the integration fixture places ground objects at
known world coordinates so the expected render pixel is a closed-form expression.

The exception is `golden/`, which is deliberately the opposite: whole-image fixtures captured
from the ModernGL renderer before the multi-backend refactor. Derived expectations check that
a render is *correct*; the goldens check that it has not *changed*, which is the only thing
that catches a silent shift in orientation, fill rule or channel order. Re-bless them with
`python -m test.golden.capture`, and review the diff when you do.

Image comparisons go through `helpers/images.assert_images_close`, which reports MAE, worst
pixel and the fraction out of tolerance — never a bare `array_equal`, which would make the
suite useless across platforms and torch builds.

## The tests that matter most

If you change the rasteriser, these are the ones that catch you:

- `test_shared_edges_are_covered_exactly_once` — the top-left fill rule. Without it every
  internal DEM edge is double-counted, so the integral normalises those pixels by the wrong
  number of shots.
- `test_perspective_correct_barycentrics` — interpolation weights divided by `w`. A naive
  screen-space linear interpolation passes every orthographic test and fails only here.
- `test_integral_alpha_counts_overlapping_shots` — three identical shots must accumulate
  exactly 3.0. Not 6 (double-counted edges), not 2 (dropped fragments).
- `test_repeated_renders_are_bit_identical` — renders must depend only on their inputs. The
  direct regression guard for the intermittent-artifact class; it caught a real
  uninitialised-memory bug.
- `test_coverage_is_independent_of_pixel_values` (`backends/`) — a shot whose fourth channel
  is zero still counts as one observation. Under the old alpha-as-counter scheme it counted
  as none, which is what cost the embedded-light-field prototype a quarter of its channels.
- `test_every_channel_survives` (`backends/`) — channel *i* holds the constant *i*, so every
  output channel must still read *i*. Catches a channel being overwritten by the coverage
  counter.
- `test_projected_shot_matches_analytic_uv_mapping` — the whole chain at once: matrix
  convention, perspective-correct interpolation, texture orientation, framebuffer row order.

## Regression tests for fixed defects

`unit/test_convert_regressions.py` and `integration/test_label_camera_agreement.py` guard the
defects fixed during the PyTorch port (`docs/MIGRATION_REVIEW.md` §9). They used to be
*characterisation* tests pinning wrong-but-load-bearing behaviour; now that the behaviour is
correct, they assert the fix and keep the history in their docstrings.

The one worth understanding: `pixel_to_world_coord` used a transposed rotation, and
`ProjectionScene._camera_for_frame` negated its Euler angles to compensate. That cancellation
only held for single-axis rotations, so a genuinely tilted pose mis-projected labels by
metres. Both halves were fixed together, and
`test_label_camera_agreement.py` renders a marker through tilted/rolled/yawed poses to prove
the label lands on it. Its fixture uses a deliberately steep pose — a 10° tilt produces only
~0.5 px of error, which would slip under any sane tolerance and make the test decorative.

## Cross-engine equivalence

Every backend must render the same picture, and since the merge that comparison happens
**in one process**: `test/backends/test_cross_engine.py` renders all five golden cases
through every available backend and compares pixels and coverage. Measured ModernGL vs
PyTorch: mean absolute error 0.24–0.85 of 255, 0.4–1.3 % of values off by more than 8,
coverage differing on at most 0.01 % of pixels. ModernGL vs Vulkan is essentially bit-exact.

That comparison used to require a subprocess, because the two projects each installed a
package called `alfspy` and could not share an interpreter. `test/parity/` retains that
harness for comparing against a separate **2.x** checkout:

```bash
export ALFS_REFERENCE_PATH=/path/to/alfs_py     # set ALFS_REFERENCE_PATH=C:\D\Projects\alfs_py
pytest test/parity -m parity -v
```

The reference renders in a **subprocess** under the other checkout's interpreter (found at
`$ALFS_REFERENCE_PATH/.venv`, or set `ALFS_REFERENCE_PYTHON`), because both install a package
called `alfspy` and cannot share a process. Scene inputs and results are exchanged as `.npy`
files. For comparing backends *within* this project, use `test/backends/` instead.

Thresholds: mean absolute error ≤ 1/255 and at most 2 % of pixels off by more than one LSB.
Exact equality is not achievable — GL quantises depth to 24 bits and snaps vertices to a
fixed-point subpixel grid — so parity is defined perceptually.

**Measured** (Intel Arc, Windows, ModernGL 5.8.2):

| scene | MAE | >1 LSB |
|---|---|---|
| background render | 0.115 | 0.000 % |
| projected shot, flat DEM | 0.246 | 0.586 % |
| projected shot, undulating DEM | 0.335 | 0.453 % |
| integral of 4 shots | 0.349 | 0.943 % |

Disagreements are confined to coverage boundaries; interiors match essentially exactly.

## Markers

- `slow` — production-resolution renders.
- `gl` — needs a working OpenGL context; skipped when none can be created.
- `cuda` — skipped automatically without a CUDA device.
- `parity` — skipped automatically without `ALFS_REFERENCE_PATH`.
