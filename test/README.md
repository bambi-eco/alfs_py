# Test suite

Runs on CPU with no GPU, no OpenGL driver and no virtual framebuffer — which is the point of
the migration.

```bash
pip install -r requirements.txt
pip install pytest
pytest -q                       # everything (~2 s)
pytest test/raster -v           # the rasteriser proofs
pytest -m "not slow" -q
```

## Layout

| Directory | What it proves | Needs |
|---|---|---|
| `unit/` | Pure functions: geometry, cameras, coordinate conversion, labels, image helpers. No rendering. | — |
| `raster/` | The torch rasteriser against **independently derived** answers. This is where the ModernGL replacement is actually proved. | — |
| `integration/` | `ProjectionScene` end to end on a generated synthetic flight. | — |
| `parity/` | Comparison against a real ModernGL `alfs_py`. **The acceptance gate.** | `ALFS_REFERENCE_PATH` |
| `bench/` | Throughput measurements. Not collected by pytest. | — |
| `tools/` | `probe_pyrr` — regenerates the matrix-convention table in `docs/`. | — |
| `helpers/` | Synthetic scenes, the flight-dataset generator, tolerance-based image comparison. | — |

## What makes these tests worth trusting

Expected values are **derived, not recorded**. `test/raster/test_rasterizer.py` builds its
coverage masks with plain numpy from the camera matrices; `test_renderer.py` recomputes the
projective UV mapping from pinhole geometry; the integration fixture places ground objects at
known world coordinates so the expected render pixel is a closed-form expression. There are
no golden images to re-bless when something changes.

Image comparisons go through `helpers/images.assert_images_close`, which reports MAE, worst
pixel and the fraction out of tolerance — never a bare `array_equal`, which would make the
suite useless across platforms and torch builds.

## The tests that matter most

If you change the rasteriser, these are the ones that catch you:

- `test_shared_edges_are_covered_exactly_once` — the top-left fill rule. Without it every
  internal DEM edge is double-counted, and since alpha is the ALFS overlap counter the
  integral normalises those pixels by the wrong number of shots.
- `test_perspective_correct_barycentrics` — interpolation weights divided by `w`. A naive
  screen-space linear interpolation passes every orthographic test and fails only here.
- `test_integral_alpha_counts_overlapping_shots` — three identical shots must accumulate
  exactly 3.0. Not 6 (double-counted edges), not 2 (dropped fragments).
- `test_repeated_renders_are_bit_identical` — renders must depend only on their inputs. This
  is the direct regression guard for the artifact class the migration exists to remove; it
  caught a real uninitialised-memory bug carried over from `alfs_py`.
- `test_projected_shot_matches_analytic_uv_mapping` — the whole chain at once: matrix
  convention, perspective-correct interpolation, texture orientation, framebuffer row order.

## Regression tests for fixed defects

`unit/test_convert_regressions.py` and `integration/test_label_camera_agreement.py` guard the
defects fixed during the migration (`docs/MIGRATION_REVIEW.md` §9). They used to be
*characterisation* tests pinning wrong-but-load-bearing behaviour; now that the behaviour is
correct in both projects, they assert the fix and keep the history in their docstrings.

The one worth understanding: `pixel_to_world_coord` used a transposed rotation, and
`ProjectionScene._camera_for_frame` negated its Euler angles to compensate. That cancellation
only held for single-axis rotations, so a genuinely tilted pose mis-projected labels by
metres. Both halves were fixed together, and
`test_label_camera_agreement.py` renders a marker through tilted/rolled/yawed poses to prove
the label lands on it. Its fixture uses a deliberately steep pose — a 10° tilt produces only
~0.5 px of error, which would slip under any sane tolerance and make the test decorative.

## Parity against ModernGL

The port is only "equivalent" if it matches the reference renderer. CI cannot check that — no
GL driver — so run it on a workstation where the old stack renders correctly (Windows or
on-prem Linux, *not* the Xvfb container whose artifacts prompted this work):

```bash
export ALFS_REFERENCE_PATH=/path/to/alfs_py     # set ALFS_REFERENCE_PATH=C:\D\Projects\alfs_py
pytest test/parity -m parity -v
```

The reference renders in a **subprocess** under `alfs_py`'s own interpreter (found at
`$ALFS_REFERENCE_PATH/.venv`, or set `ALFS_REFERENCE_PYTHON`). Both projects install a
package called `alfspy`, so they cannot share a process, and each needs its own dependency
set. Scene inputs and results are exchanged as `.npy` files.

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
- `cuda` — skipped automatically without a CUDA device.
- `parity` — skipped automatically without `ALFS_REFERENCE_PATH`.
