# ALFSPy → ALFSTorch: Migration Review

Review of `C:\D\Projects\alfs_py` (ModernGL) and design of the PyTorch replacement in
`C:\D\Projects\alfs_pytorch`.

---

## 1. Inventory: where the GPU actually lives

The codebase is ~70 source files, but the OpenGL dependency is remarkably **shallow**. Only
seven files import ModernGL for real work; the rest import it only for type annotations or
receive a `Context` object they pass straight through.

| File | Nature of GL use | Migration cost |
|---|---|---|
| `core/rendering/renderer.py` | **Real.** Owns both GLSL programs, the FBO, blending state, all draw calls. | Rewrite |
| `core/rendering/data.py` | **Real.** `RenderObject` wraps VAO/VBO/IBO/Texture; `TextureData.to_bytes` encodes for GL upload. | Rewrite |
| `core/rendering/shot.py` | **Real.** `CtxShot` owns a GL `Texture` per shot, lazy upload/release. | Rewrite (thin) |
| `core/util/moderngl.py` | **Real.** `img_from_fbo` — FBO readback + vertical flip. | Rewrite (5 lines) |
| `render/render.py` | `make_mgl_context()` + type hints. | Mechanical |
| `render/projection.py` | Context creation / type hints. | Mechanical |
| `alfs.py`, `alfs2.py`, `animation.py`, `create_labels.py`, `orthograf(ph)ic_projection.py`, `test.py` | Type hints and one `make_mgl_context()` call. | Mechanical |
| `render/render_sp.py`, `core/sharepoint/` | SharePoint integration. | **Removed** (§10) |

**Everything else is already pure NumPy/OpenCV/trimesh/pyrr** — `core/geo`, `core/convert`,
`core/util` (minus `moderngl.py`), `render/data.py`, the label/YOLO logic. That code is
copied verbatim apart from the defect fixes listed in §9.

`resources/shaders/*.glsl` is **dead**: the renderer embeds its shaders as f-strings in
`renderer.py`. The `.glsl` files are kept for reference but nothing loads them.

### Duplicate entry points

`orthografic_projection.py` (36 KB, the Dockerfile entry point) and
`orthographic_projection.py` (27 KB) are near-duplicate pipelines, and `alfs.py` / `alfs2.py`
overlap heavily with both. `render/projection.py` is the newest and cleanest surface — a
`ProjectionScene` context manager wrapping the whole thing. **This review does not
consolidate them** (that would change behaviour beyond the port), but it flags them: the test
pipeline targets `ProjectionScene` and `Renderer` as the contract, so a later consolidation
has a safety net.

---

## 2. What the renderer actually computes

Two GLSL programs, both trivially expressible as tensor math.

### 2.1 `OBJ` program — textured background

Standard forward render of the DEM mesh with its baked ortho-photo texture:

```
gl_Position = P · V · M · v
color       = texture(u_s2_tex, uv)
```

Depth test on, back-face culling on, clear to `TRANSPARENT = (0,0,0,0)`.

### 2.2 `SHOT` program — projective texture mapping

This is the heart of both orthographic projection and ALFS. The mesh is rasterised from the
**virtual** camera, and each fragment looks *back* into the source frame:

```glsl
// vertex
world_pos          = M · v
gl_Position        = P_virtual · V_virtual · world_pos
v_out_v4_shot_uv   = P_shot · C_shot · V_shot · world_pos

// fragment
uv  = shot_uv.xyz / shot_uv.w / 2 + 0.5      // NDC → [0,1]
if (uv.w <= 0 || uv.x∉[0,1] || uv.y∉[0,1]) discard;
color = texture(u_s2_tex, uv.xy);
if (u_f_mask > 0) color *= texture(u_s2_mask, uv.xy).r;
```

Three things to note, because they are easy to get wrong in a port:

1. **`uv.w` after the rewrite is always `1.0`.** The guard `uv.w <= 0.0` is therefore dead
   code in the original shader — points *behind* the shot camera are **not** rejected by it.
   They are rejected only if their post-division `x`/`y` fall outside `[0,1]`, which is not
   guaranteed. This is a latent correctness bug in the original. The port reproduces the
   original behaviour by default and exposes `reject_behind_camera=True` to fix it (see §6).
2. **`P·C·V`, not `P·V·C`.** The correction matrix sits between projection and view, and
   `CtxShot.get_correction()` returns `T⁻¹ · R` (inverse translation, forward rotation).
   Order matters; the tests pin it.
3. **The interpolated quantity is the clip-space vector, not the UV.** Interpolation happens
   *before* the perspective divide, so the port must interpolate `shot_uv` perspective-
   correctly in clip space and divide per fragment — not interpolate the final UVs.

### 2.3 Integration (`render_integral`)

```
enable(BLEND); disable(DEPTH_TEST); blend_func = ADDITIVE
clear(0,0,0,0)
for shot in shots: draw(mesh, SHOT program)
read FBO as float32
alpha = accumulated.a                       # == number of shots covering the pixel
out   = accumulated / alpha  where alpha > alpha_threshold
optional auto-contrast over RGB of the covered region
result = (out*255).astype(uint8)[::-1]
```

The alpha channel doubles as an **overlap counter** — each contributing shot adds `1.0`
(source images are RGBA with `a=255→1.0`). That is the whole ALFS integral.

**`DEPTH_TEST` is disabled during integration.** With back-face culling still on and a
height-field DEM viewed from above this is *almost* equivalent to depth-testing, but a true
overhang double-counts. The port makes this explicit and configurable (§6).

---

## 3. Root cause of the Docker artifacts

The README blames "incorrectly cleaned buffers … when having Alpha values next to the
colors … sometimes (not always)". That is consistent with what the code does:

* `Renderer.__init__` creates the FBO with `dtype='f4'` and `components=4`, then calls
  `self._fbo.use()` **once**. Nothing ever rebinds it.
* `render_background()` and `_psi_*` clear via `self._ctx.clear(...)` — a **context-level**
  clear that targets whatever framebuffer is currently bound — while `render_integral()`
  clears via `self._fbo.clear(...)`. The two paths disagree about which surface they are
  clearing.
* Under Xvfb the software GL stack (llvmpipe/swrast) provides a default framebuffer, and
  binding of the standalone FBO is not guaranteed to survive across the driver's own
  state changes. A context-level clear can then hit the *window* framebuffer while draws go
  to the FBO — leaving the previous frame's accumulated RGBA in place. Because the alpha
  channel is a *counter*, stale alpha does not look like stale transparency: it silently
  divides the accumulated colour by the wrong overlap count, which is exactly the
  "sometimes, not always" artifact reported.
* `mgl.ADDITIVE_BLENDING` is `(ONE, ONE)` applied to RGB **and** alpha, so any leftover
  garbage in the buffer is permanently baked into the normalisation.

There is no software fix for this inside ModernGL that does not amount to "hope the driver
cooperates". Removing the GL context removes the failure mode entirely: in the torch backend
the accumulator is a plain tensor allocated per render call, and "clear" is `zeros_()`.

### 3.1 A second, host-side cause — found during the port

`render_integral` normalises with:

```python
out = np.divide(integral_arr, alpha, where=alpha_mask)      # upstream
```

**`np.divide` with `where=` and no `out=` leaves the excluded elements uninitialised.** numpy
allocates the output buffer and only writes where the mask is true; everywhere else the array
holds whatever was in that heap memory. Those values then go through `(out * 255).astype(np.uint8)`,
which is undefined behaviour for NaN and overflows for large floats.

So *every pixel below `alpha_threshold`* — i.e. the entire region no shot covered — was filled
with recycled memory rather than zeros. This reproduces the reported symptom precisely:

* it is host-side, so it happens **in Docker and on-premise alike** — but it is normally
  invisible on-premise because `ADD_BACKGROUND=1` paints the DEM render over the uncovered
  region, and the Docker ALFS runs are exactly where that region is largest;
* it is "sometimes, not always", because it depends on what the allocator hands back;
* it affects the alpha channel too, which is why the artifacts look like transparency errors.

The port passes an explicit `out=np.zeros_like(...)` in both `render_integral` and
`project_shots(integral=True)`. This was caught by
`test/raster/test_renderer.py::test_repeated_renders_are_bit_identical`, which renders the
same scene twice with unrelated work in between and compares bit-for-bit — it failed on the
faithfully ported code and passes now.

**Recommendation: apply this one-line fix to `alfs_py` as well.** It is independent of the
rendering backend, and if the Xvfb artifacts persist after switching to torch, this was
probably the whole story.

---

## 4. Replacement design

A small `alfspy.core.torchgl` package provides exactly the primitives the renderer needs — no
attempt to reimplement OpenGL.

```
core/torchgl/
  context.py    TorchContext      device/dtype + GL-shaped state flags
  texture.py    TorchTexture      NCHW float tensor + bilinear sampler
  framebuffer.py TorchFramebuffer  H×W×4 float accumulator, clear/read
  raster.py     rasterize()       depth-buffered triangle rasteriser
  programs.py   the two shaders as torch functions
```

### 4.1 Rasteriser

The one non-trivial piece. Requirements: 2048×2048 output, DEM meshes with 10⁵–10⁶ triangles,
must run on CPU *and* CUDA, must be deterministic.

**Algorithm** — bucketed binning with a packed depth/ID key:

1. **Transform.** `clip = (P·V·M) · v` for all vertices at once. Screen coords
   `s = ((ndc.xy * ±1) + 1)/2 * (W,H)`, `z_ndc = clip.z/clip.w`, `inv_w = 1/clip.w`.
2. **Cull.** Reject triangles that (a) have any vertex with `clip.w <= ε` (near-plane
   crossing — see §6), (b) fail the signed-area back-face test when culling is enabled,
   (c) have a screen bounding box fully outside the viewport or of zero area.
3. **Bin.** Bucket surviving triangles by `2^ceil(log2(max(bbox_w, bbox_h)))`. Each bucket
   uses a fixed `s×s` candidate grid anchored at the triangle's bbox origin, so the
   candidate expansion is a single broadcast per bucket instead of a ragged gather. Buckets
   are processed in sample-budget-limited chunks so peak memory is bounded regardless of
   mesh size or how pathological the triangle size distribution is.
4. **Cover.** Per candidate pixel centre, three edge functions give barycentrics; inside test
   is `all(λ ≥ 0)` with a top-left-ish tie rule folded into the ε.
5. **Resolve.** Depth `z` is quantised to uint32 and packed with the triangle index into one
   int64 key: `key = (z_q << 32) | tri_id`. A single `scatter_reduce_(..., "amin")` over the
   flat pixel index resolves *both* the nearest fragment and the tie-break (lowest triangle
   index) in one deterministic pass — no sorting, no atomics race, identical on CPU and CUDA.
   32-bit depth quantisation is *finer* than a typical 24-bit GL depth buffer.
6. **Shade.** Unpack the winning `tri_id` per covered pixel, recompute barycentrics for those
   pixels only, apply perspective correction
   `λ̂ᵢ = λᵢ·inv_wᵢ / Σⱼ λⱼ·inv_wⱼ`, interpolate attributes, run the program.

When depth testing is *disabled* (the integral path), step 5 is skipped and every inside
fragment is `scatter_add`-ed into the accumulator directly — matching `disable(DEPTH_TEST) +
ADDITIVE_BLENDING` exactly.

**Complexity** is O(covered samples), not O(triangles × pixels). For a top-down DEM the
covered-sample count is ≈ the output pixel count times a small constant.

**The fill rule is not optional.** GL's top-left rule decides which of two triangles owns a
pixel centre lying exactly on their shared edge. Skipping it makes both claim the pixel — 
harmless for an opaque depth-tested draw, but fatal for ALFS, where alpha is the *overlap
counter*: every internal DEM edge would report one extra contributing shot and normalise to
the wrong colour. That is a lattice of seams across the whole integral.

Getting the rule to work in floating point needs one more step. Two triangles traverse a
shared edge in opposite directions, so each evaluates the edge function from a *different*
anchor vertex. Those expressions are exact negatives in real arithmetic but not in float:
both can round to a small positive value and both triangles claim the pixel anyway. Hardware
rasterisers avoid this by snapping vertices to a fixed-point subpixel grid. The port instead
anchors every edge at its **lexicographically smaller endpoint**, so both triangles compute
the identical float and the sign flip that restores each triangle's orientation is exact.
This is cheap (precomputed per triangle in `setup_triangles`, not per sample) and stays in
float32.

Both halves are pinned by tests: `test_shared_edges_are_covered_exactly_once` (rasteriser
level) and `test_dem_mesh_renders_without_seams` / `test_integral_alpha_counts_overlapping_shots`
(renderer level). All three failed before the fix.

### 4.1.1 Rasterisation is cached across shots

Within one `render_integral` call the mesh, the virtual camera and the render state are
constant — only the shot texture and its matrices change. Coverage is therefore identical for
every shot, and the renderer rasterises **once** and shades N times. `apply_matrices`
invalidates the cache, so moving the camera (as `animate_focus` does) still re-rasterises.

Measured on this machine (CPU, torch 2.13, 2048×2048 output, 131 072-triangle DEM):

| | before caching | after |
|---|---|---|
| single shot projection | 823 ms | 256 ms |
| integral of 8 shots | 5 535 ms | 2 094 ms |
| integral of 30 shots (a realistic ALFS window) | ~20 s | 6 950 ms |

At 1024×1024 the 8-shot integral is 445 ms. See §8.1 for what to do if that is still too slow.

### 4.2 Texture sampling

GL `texture()` with default state = bilinear filtering, `GL_REPEAT` wrap, `v=0` at the
*bottom*. `torch.nn.functional.grid_sample(mode='bilinear', align_corners=False)` matches
GL's bilinear filter exactly once UVs are mapped `[0,1] → [-1,1]`. Textures are stored
bottom-up (as `TextureData.to_bytes()` already does for GL), so UV semantics are unchanged
and every downstream `[::-1]` flip in the existing code stays correct.

### 4.3 API preservation

`Renderer`, `CtxShot`, `TextureData`, `MeshData`, `Resolution`, `RenderResultMode`,
`ShotLoader` and `ProjectionScene` keep **identical constructor signatures, method names and
return types**. `make_mgl_context()` is kept as a deprecated alias of `make_torch_context()`,
so downstream scripts (`alfs.py`, the two projection entry points, `create_labels.py`) need
only an import swap.

The `ctx` parameter threaded through everything becomes a `TorchContext`. It carries the
device and dtype, which is how CPU/GPU selection propagates without touching call sites.

Signatures that did change — all backend-internal, none on the public rendering path:

| Symbol | Change | Why |
|---|---|---|
| `RenderObject.from_mesh` | `(prog, mesh, texture, vert_par, uv_par)` → `(ctx, mesh, texture)` | The shader-attribute names existed only to build a VAO. There is no VAO. |
| `RenderObject` fields | VAO/VBO/IBO handles → tensors (`vertices`, `indices`, `uvs`, `tex`) | Same reason; field names preserved where meaningful. |
| `CtxShot.tex_use(loc)` | still present; new `get_texture(ctx)` returns the texture | Textures are passed to the shading functions explicitly instead of through global texture units. |
| `Renderer.render_integral(save_name=...)` | type hint `Iterator[str]` → `str` | The annotation was wrong upstream; every caller already passed a string. |
| `ProjectionScene(gl_backend=...)` | accepted and ignored, documented as deprecated | This is the parameter the Xvfb deployment needed. There is no GL backend to choose. |
| `ProjectionScene(device=...)` | new | Selects the torch device without constructing a context by hand. |
| `alfspy.core.util.moderngl` | → `alfspy.core.util.framebuffer` | Module named after the dependency being removed. `img_from_fbo` is unchanged. |
| `src/alfspy/test.py` | → `src/alfspy/scratch.py` | Collided with pytest collection. It did not import in `alfs_py` either; see its docstring. |
| `alfspy.render.render_integral_sp` | **removed** | SharePoint support dropped; see §10. |
| `ProjectionScene._camera_for_frame` | rebuilt from the renderer's own matrices | §9.3. |

---

## 5. Version-compatibility policy

The user's constraint: *do not pin torch tightly.*

* **`torch>=1.13`**, no upper bound. The rasteriser uses only: `einsum`/`matmul`, advanced
  indexing, `grid_sample`, `scatter_add_`, and `Tensor.scatter_reduce_`.
  `scatter_reduce_` landed in 1.12 and stabilised in 1.13 — that sets the floor.
* A **runtime capability probe** (`torchgl/compat.py`) checks `scatter_reduce_` at import and
  falls back to a sort-based resolve if it is missing or behaves differently, so a future
  signature change degrades rather than breaks.
* No `torch.compile`, no `torch.func`, no `functorch`, no custom CUDA kernels, no
  `torch.library` — the parts of the API that churn most.
* `numpy` unpinned above `1.23` and **NumPy 2.x compatible** (the original pinned
  `numpy~=1.23.5`, which blocks torch ≥ 2.3 wheels built against NumPy 2).
* `requires-python >= 3.9` retained; the code already uses PEP-585 generics (`tuple[int,int]`)
  which is 3.9+.
* `trimesh` unpinned within `>=3.21,<5` — the original had a `3.21.7` pin in `pyproject.toml`
  fighting a `4.6.6` pin in `requirements.txt`. That conflict is resolved in favour of the
  4.x line, with the ray backend selected at runtime (`embreex` if present, else the pure
  Python backend).

Rationale: the *only* place a torch version bump can plausibly break this port is
`scatter_reduce_` semantics and `grid_sample` corner behaviour. Both are covered by dedicated
unit tests that fail loudly with an actionable message rather than silently changing renders.

---

## 6. Deliberate divergences from the ModernGL version

These are the places where "faithful port" and "correct" disagree. All are **off by default**
so the port is bit-comparable to the reference, and all are covered by tests.

| Flag (on `TorchContext` / `Renderer`) | Default | Effect when enabled |
|---|---|---|
| `reject_behind_camera` | `False` | Rejects fragments with `shot_clip.w <= 0` — fixes the dead `uv.w <= 0` guard in the original shader (§2.2). |
| `depth_test_in_integral` | `False` | Depth-resolves each shot's draw before accumulating, so an overhanging DEM triangle contributes once instead of twice (§2.3). |
| `clip_near_plane` | `False` | Properly clips triangles straddling the near plane instead of discarding them whole. Only matters for perspective virtual cameras at grazing angles; the ortho path never triggers it. |

Two divergences are **not** optional because they are unconditional improvements:

* **No stale-buffer failure mode.** Accumulators are freshly allocated per call.
* **float64 accumulation option.** `project_shots(integral=True)` in the original accumulates
  into `np.uint64` after an implicit float→uint8 round-trip per shot, which quantises every
  contribution to 8 bits *before* summing. The torch path accumulates in float32 (or float64
  via `TorchContext(dtype=torch.float64)`). The reference behaviour is reproducible with
  `quantize_per_shot=True` for comparison testing.

---

## 7. Test strategy

> **Status: implemented.** 150 tests pass, 4 skip (3 parity + 1 CUDA) in ~2 s on CPU with no
> GPU, no GL driver and no display. Run with `pytest -q`.
>
> Verified end to end against **Python 3.10, torch 2.13.0, numpy 2.2.6, trimesh 5.0.0,
> opencv 5.0.0** — every one of them newer than anything the upstream pins allowed, which is
> the loose-pinning policy of §5 doing its job.
>
> Several defects were found *by these tests*. One — shared-edge double counting — was
> introduced by the new rasteriser and fixed before merge (§4.1.1). The rest were
> pre-existing in `alfs_py` and are now fixed in **both** projects; see §9.

Three tiers, all runnable without a GPU and without any real flight data.

**Tier 1 — unit (`test/unit/`), no rendering.**
Pure functions: `change_pixel_origin`, `adjacent_angle`, `quaternion_from_eulers`,
`Transform`/`Frustum`/`Camera` matrices against closed-form pyrr results, `AABB`, `overlay`,
`gen_checkerboard_tex`, YOLO conversion, MOT parsing, correction-JSON loading.

**Tier 2 — rasteriser (`test/raster/`), synthetic geometry with analytic ground truth.**
This is where the ModernGL replacement is actually *proved*:
* A full-screen quad renders a known texture back exactly (identity projection) — bit-level.
* A single triangle's coverage matches an independently computed analytic mask to ≤1 pixel
  on the boundary.
* Depth ordering: two overlapping quads at different depths — the near one wins everywhere,
  and swapping draw order changes nothing (depth test correctness, not draw-order luck).
* Back-face culling: a wound-backwards triangle produces zero coverage.
* Perspective-correct interpolation: a texture-mapped quad at 60° to the camera matches the
  analytic perspective-correct UVs (this is the test that catches naive linear interpolation).
* Projective texture mapping: place a shot camera and a virtual camera such that the mapping
  is an exact known homography on a flat DEM; compare against `cv2.warpPerspective`.
* Additive integration: N identical shots ⇒ `alpha == N` exactly, normalised RGB == the
  single-shot RGB.
* `scatter_reduce_`/`grid_sample` behaviour probes with explicit failure messages.

**Tier 3 — integration (`test/integration/`), end-to-end on a generated dataset.**
A fixture synthesises a complete miniature flight: a procedural DEM (a `.glb` with a baked
texture), a matched-poses JSON, a correction JSON, a mask, N frames rendered *from known
camera poses* with a known pattern, and MOT labels for known ground objects. Then:
* `ProjectionScene.project_orthographic` on a frame reproduces the pattern at the expected
  world scale (metres-per-pixel checked against `ortho_size / resolution`).
* Labels round-trip: pixel → world → render-pixel lands on the object we planted, within a
  few pixels.
* `render_lightfield` over the synthetic flight produces alpha == the true overlap count per
  pixel, and merged track labels enclose the per-frame boxes.
* Determinism: the same render twice is bit-identical; CPU and CUDA agree to ≤1 LSB (skipped
  when no CUDA).
* Release/leak: repeated scene create/release does not grow allocated tensors.

**Reference-parity harness (`test/parity/`, opt-in).**
Marked `@pytest.mark.parity`, skipped unless `ALFS_REFERENCE_PATH` points at a working
ModernGL `alfs_py`. Runs the same synthetic scene through both stacks and asserts
mean-absolute-error ≤ 1/255 and ≥98 % of pixels within ±1 LSB. This is the acceptance gate
for "the port is equivalent"; it deliberately does *not* run in CI, where no GL is available.
Measured results in §7.1.

Image comparison uses a tolerance helper (`assert_images_close`) reporting MAE, max
deviation, and the fraction of pixels beyond tolerance — never a bare `array_equal`, which
would make the suite useless across platforms.

---

## 7.1 Parity result — measured

The gate has been **run**, not just written. Against `alfs_py` on real hardware
(Intel Arc, Windows, ModernGL 5.8.2):

| scene | MAE | RMSE | max | pixels off by >1 LSB |
|---|---|---|---|---|
| background render, undulating DEM | 0.115 | 0.34 | 1 | 0.000 % |
| projected shot, flat DEM | 0.246 | 3.22 | 42 | 0.586 % |
| projected shot, undulating DEM | 0.335 | 5.60 | 255 | 0.453 % |
| integral of 4 overlapping shots | 0.349 | 4.71 | 125 | 0.943 % |

Mean absolute error is below one 8-bit level everywhere, and over 99 % of pixels match
within one level. The disagreements are concentrated on **coverage boundaries** — the
footprint edge and DEM silhouettes — which is expected: a hardware rasteriser snaps vertices
to a fixed-point subpixel grid, so a pixel centre within a fraction of a level of an edge can
legitimately fall either way. Interiors agree essentially exactly, as the background row
shows.

Reproduce with:

```
set ALFS_REFERENCE_PATH=C:\D\Projects\alfs_py
pytest test/parity -m parity -v
```

The reference render runs in a **subprocess** under `alfs_py`'s own interpreter: both
projects install a package called `alfspy` and cannot share a process, and each needs its own
dependencies (ModernGL there, torch here). Scene inputs and results are exchanged as `.npy`.

## 8. Risks and open items

1. **Rasteriser throughput — the main open risk.** Measured on CPU (see §4.1.1): a realistic
   30-shot ALFS integral at 2048×2048 over a 131 k-triangle DEM takes **~7 s per output
   frame**, of which ~230 ms per shot is *shading* (interpolating the shot varying for ~4 M
   fragments, then `grid_sample`). Geometry is no longer the bottleneck after caching.

   For a dataset of a few thousand frames that is hours of CPU time. Options, cheapest first:
   * **Run on CUDA.** Everything is device-agnostic already; pass `device='cuda'` to
     `ProjectionScene` or `make_torch_context`. Untested here (no GPU on this machine) — the
     `cuda`-marked tests will verify parity when run on one.
   * **Lower the render resolution.** Cost is roughly linear in output pixels.
   * **Drop to `nvdiffrast`** behind the same `rasterize()` signature — the interface was
     designed to allow it.

   Benchmark harness included: `python -m test.bench.bench_raster --device cuda`. It is
   excluded from the default test run.
2. **Very large DEM textures.** `MAX_TEX_DIM = 8192` and the `CPP_INT_MAX` downscale in
   `process_render_data` exist purely because ModernGL's C++ layer uses `int` for byte counts.
   Torch has no such limit; the caps are kept (so renders match) but are now
   *configurable* rather than hard-coded.
3. **`orthografic_projection.py` vs `orthographic_projection.py`.** Both ported, both kept.
   Consolidation is a follow-up, now testable.
---

## 9. Defects found and fixed

These were all latent in `alfs_py`. The first review pinned them with characterisation tests
and left the behaviour alone; they have since been **fixed in both projects** and the
characterisation tests were promoted to regression tests
(`test/unit/test_convert_regressions.py`, `test/integration/test_label_camera_agreement.py`).

### 9.1 Uninitialised `np.divide` output — the artifact bug

Covered in detail in §3.1. `np.divide(..., where=mask)` without `out=` left every pixel below
the alpha threshold holding uninitialised heap memory. Fixed in `render_integral` and
`project_shots(integral=True)` in **both** projects, along with an `auto_contrast`
divide-by-zero guard for uniform regions.

### 9.2 Non-deterministic GLTF buffer resolution

Covered in §4.4. `gltf.resources[0].data` assumed the geometry buffer is the first resource;
gltflib's ordering is not deterministic once a texture is also embedded as a data URI, so the
reader could decode the PNG as vertex floats. Buffers are now resolved by their own URI, and
`gltf_to_texture_data` goes through `textures[i].source` instead of indexing `images` by
texture index. Fixed in both projects.

### 9.3 The transposed ray rotation, and its compensator

`pixel_to_world_coord` built rays with `np.dot(dirs, R33.T)` — the inverse rotation, so the
centre ray pointed opposite to `camera.transform.forward`. This was cancelled by
`ProjectionScene._camera_for_frame` negating every Euler angle.

The cancellation was **only approximate**: `R(-e) == R(e).T` holds for a single-axis rotation
but not for composed Euler angles. Near-nadir, yaw-only BAMBI poses hid this; measured on a
`(22°, 14°, 125°)` pose with a non-neutral correction, the old label camera's view matrix was
off by 46 units and its centre ray landed **3.0 m** from the truth on the ground.

Both halves are now fixed:
* rays use `R33`, not its transpose;
* `_camera_for_frame` reconstructs the label camera from the *same* matrices the renderer
  uses — `V_shot @ C`, exactly as the shot program applies them — via a new
  `_camera_from_view()` helper. Rotation construction is shared with `_make_shot` through
  `_rotation_for()`, so the two can no longer drift apart.

Guarded by `test/integration/test_label_camera_agreement.py`, which renders a marker through
tilted/rolled/yawed poses and asserts the projected label lands on it, plus a structural test
that `_camera_for_frame(i).get_view() == V_shot @ C`.

### 9.4 `world_to_pixel_coord` broadcast

`ndc_coords / ndc_coords[:, 3]` divided an `(N, 4)` array by an `(N,)` one: it raised for
every point count except 1 and 4, and at `N == 4` computed `clip[i, j] / w[j]`, dividing
every point by the *first* point's depth. Production escaped it because labels are always
four corners and the render camera is always orthographic (all `w == 1`). Now `[:, 3:4]`,
so any point count works and perspective cameras are correct. Also computes in float64.

### 9.5 The orthographic branch of `pixel_to_world_coord`

Carried a standing `# TODO: fix and test` and had never run: it evaluated
`camera.transform.rotation @ origin_offset`, multiplying a 4-element quaternion by a `(3, N)`
array, and divided the already normalised NDC offset by `width`/`height` a second time.
Now builds an `(N, 3)` offset, scales by half the frustum size, and rotates with `@ R33`.

### 9.6 `change_pixel_origin` was not invertible

It translated first and negated afterwards, so converting to an origin whose axes point the
other way and back left a residual of `2 · max · offset`. Only `TopLeft` and `TopCenter`
round-tripped. It now routes through top-left coordinates and is exact for all 81 origin
pairs (exhaustively tested).

### 9.7 `world_to_pixel_coord2`

Used `(P_world − t) @ R` where the row-vector convention needs `@ R.T`, and its orthographic
branch scaled NDC by `width`/`height` instead of by half the frustum size. Both corrected.
Still unused by any pipeline.

---

## 10. SharePoint support removed

`core/sharepoint/`, `render/render_sp.py`, `run_sp()` and the
`Office365-REST-Python-Client` dependency are gone from **both** projects.

Beyond being unused for the rendering work, it was actively harmful: `render/__init__.py`
imported `render_sp` eagerly, so a missing or incompatible client made the *entire* rendering
package unimportable. Current releases of that client require Python ≥ 3.11
(`from typing import Self`), which broke the 3.9/3.10 targets outright.

`glb_extract_from_bytes` and `gltf_extract_from_bytes` in `core/util/gltf.py` were kept —
they are generic byte-level loaders, not SharePoint-specific.
