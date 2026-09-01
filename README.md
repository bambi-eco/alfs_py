# AlfsPy — Airborne Light Field Sampling and Orthographic Projection

A Python framework for Airborne Light-Field Sampling (ALFS) and orthographic projection of
geo-referenced drone footage, with **three interchangeable render backends**: OpenGL
(ModernGL), a pure-PyTorch tensor rasteriser, and Vulkan (via WebGPU).

Two things distinguish version 3:

- **The renderer is pluggable.** The same code renders through OpenGL, PyTorch or Vulkan.
  Two of the three need no GL driver and no X server, so headless and containerised
  deployment does not need Xvfb.
- **A light field is not limited to three channels.** Feed the integral dense visual
  descriptors instead of colour and you get an *embedded light field*, whose every pixel
  carries a 1280-dimensional feature vector.

## Quick start

No render backend is installed by default — pick at least one, or take everything:

```bash
pip install "AlfsPy[moderngl]"    # OpenGL. Needs a GL driver.
pip install "AlfsPy[torch]"       # Tensor rasteriser. CPU or CUDA, no driver needed.
pip install "AlfsPy[vulkan]"      # Vulkan via WebGPU. Needs Python >= 3.11.

pip install "AlfsPy[all]"         # every optional feature (large: a few GB)
pip install "AlfsPy[dev]"         # all + pytest, so the suite runs without skipping
```

Extras combine, so `pip install "AlfsPy[moderngl,torch]"` gives you both engines.

```python
from alfspy import render_integral, ProjectionScene, make_context, available_engines

print(available_engines())        # ['moderngl', 'torch', 'vulkan'] — whatever works here

render_integral(dem_file, poses_file, mask_file, engine='torch')

with ProjectionScene(dem_file, poses_file, correction_file, engine='vulkan') as scene:
    scene.project_orthographic('frame_002120.png', output_image='ortho.png')
```

`$ALFS_ENGINE` sets the default when no `engine=` is given. Importing `alfspy` pulls in none
of the backends; each is imported only when selected.

```bash
pytest -q     # Tests needing an unavailable backend, device or dependency skip rather
              # than fail, so a minimal install still gives a green run.
```

### All extras

| Extra | Brings | For |
|---|---|---|
| `moderngl` | `moderngl` | the OpenGL backend |
| `torch` | `torch` | the PyTorch backend |
| `vulkan` | `wgpu` | the Vulkan backend (no-op below Python 3.11) |
| `embedding` | `torch`, `transformers`, `scikit-learn` | DINOv3 extraction, PCA reduction |
| `umap` | `umap-learn` | UMAP reduction. Separate because it drags in numba and llvmlite |
| `warp` | `warp-lang` | GPU ray casting |
| `embree` | `Rtree`, `embreex` | pins the default ray caster; **already included** (see below) |
| `all` | everything above | |
| `dev` | `all` + `pytest` | running the full suite |

The extra names match the values you pass at runtime, so `AlfsPy[torch]` goes with
`engine='torch'` and `AlfsPy[warp]` with `raycaster='warp'`. (`accel` and `raycast-gpu` are
the 3.0/3.1 names for `embree` and `warp`, kept as aliases.)

### Do I need to choose a ray caster at install time?

**Only if you want `warp`.** The default, `embree`, works out of the box: `trimesh[easy]` is
a base dependency and already pulls `embreex` and `rtree`, so a bare `pip install AlfsPy`
gives you accelerated ray casting. The `embree` extra only pins them explicitly and changes
nothing today.

That matters because the fallback is a trap rather than merely slow: without `embreex`,
trimesh silently drops to a pure-Python intersector that returns hits in a *different index
order*. `EmbreeRayCaster.accelerated` tells you which one is live.

`warp` is worth installing only if you cast far more rays than label projection does — see
[Ray casting](#ray-casting) for the measured crossover.

## Overview

The framework has two rendering modes:

1. **Orthographic Projection** — top-down, orthorectified views produced by projecting each
   frame onto a Digital Elevation Model.
2. **Airborne Light Field Sampling (ALFS)** — novel views synthesised by integrating several
   overlapping captures, without any 3D reconstruction.

Both carry 2D bounding-box labels through the same transform, which is what makes the output
usable as detection training data.

---

## Theoretical background

### Airborne Light Field Sampling

Generating novel aerial views normally means photogrammetry or a neural radiance field. ALFS
is a reconstruction-free alternative: a drone flight already captures overlapping views of
the same ground from many positions, and projective geometry is enough to combine them.

For each pixel of the output the renderer:

1. **casts a ray** from the virtual camera through the pixel into the scene,
2. **intersects** it with the DEM to find the world-space point,
3. **projects** that point back into every source image through the inverse camera matrices,
4. **samples** the colour wherever the projection lands inside the frame,
5. **averages** the samples, dividing by how many frames actually saw that point.

Step 5 is why coverage matters: a pixel seen by eight frames and a pixel seen by one must not
be averaged the same way.

### Orthographic projection

An orthographic view has no perspective divergence, so parallel lines stay parallel and a
metre is the same number of pixels everywhere in the image. That makes it suited to mapping,
size estimation, and generating detection datasets with a consistent scale.

The orthographic camera is defined by its position, its size in world units (metres) and the
output resolution.

---

## Render backends

All three implement the same three GPU operations — render the textured DEM, project a shot
onto it, integrate several shots — and produce the same picture.

| | `moderngl` | `torch` | `vulkan` |
|---|---|---|---|
| API | OpenGL 3.3 | pure tensor ops | WebGPU → Vulkan/Metal/DX12 |
| Needs a GL driver | yes | no | no |
| Headless | needs Xvfb | native | native |
| Devices | GPU | CPU or CUDA | GPU |
| Python | >= 3.9 | >= 3.9 | **>= 3.11** |

### Choosing one

```python
render_integral(dem, poses, mask, engine='vulkan')     # per call
```

```bash
export ALFS_ENGINE=torch                                # per process
```

Contexts come from **one factory whose signature is identical for every backend**, so only
the engine argument changes what you get — switching engines never means rewriting the call:

```python
from alfspy import make_context

make_context()                            # $ALFS_ENGINE, or the default
make_context('torch', device='cuda')
make_context('vulkan', device='cpu')      # software adapter
make_context('moderngl', backend='egl')   # headless Linux GL
```

Each backend honours the options that apply to it and ignores the rest, so an option meant
for one engine does not raise on another. `device` is accepted everywhere: torch uses it
directly, Vulkan maps `"cpu"` onto the software fallback adapter, and ModernGL ignores it
because OpenGL offers no device selection.

The context *is* the engine handle, so passing one selects the backend and every existing
call site keeps working unchanged:

```python
ctx = make_context('torch', device='cuda')
renderer = Renderer(resolution, ctx, camera, mesh)      # same signature as always
```

`make_mgl_context` and `make_torch_context` still work but are deprecated; they warn and
delegate to `make_context`.

`available_engines()` probes rather than guesses: `moderngl` imports successfully on a machine
with no usable GL driver and only fails when a context is created.

The engine and the device follow the same precedence — an explicit argument, then the
environment variable, then a default. So a deployment can select both without touching any
call site:

```bash
export ALFS_ENGINE=torch
export ALFS_DEVICE=cuda:1
```

`resolve_engine()` and `resolve_device()` report what a bare `make_context()` would pick.
The device default is `None` rather than a name, which leaves the choice to the backend: torch
finds CUDA on its own, and ModernGL has no device to choose.

### How closely they agree

Verified against golden fixtures captured from the OpenGL renderer:

- **Vulkan** reproduces them essentially bit for bit — the three-shot integral is exactly
  identical, the rest differ by a mean absolute error of at most 0.12 of 255 with a maximum
  deviation of 2.
- **PyTorch** differs by a mean absolute error of 0.24–0.85 of 255, with 0.4–1.3% of values
  off by more than 8 and coverage differing on at most 0.01% of pixels. It is a rasteriser
  written from scratch rather than a driver, so it breaks rasterisation ties and rounds
  texture filtering differently; disagreements are confined to coverage boundaries.

A cross-engine test suite renders every golden case through every available backend and
compares both pixels and coverage.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Flight Data    │────▶│  Pre-Processing │────▶│    Rendering    │
│  (Images + GPS) │     │  (Calibration)  │     │  (ALFS/Ortho)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   ┌─────────┐           ┌───────────┐          ┌──────────────┐
   │ Thermal │           │   DEM     │          │ Output Image │
   │  RGB    │           │   Poses   │          │    Labels    │
   │ Metadata│           │   Mask    │          └──────────────┘
   └─────────┘           └───────────┘
```

```
src/alfspy/
  core/
    rendering/     Camera, Resolution, MeshData, TextureData — no device state
    backends/      moderngl_/  torch_/  wgpu_/  + the registry
    raycast/       embree (default) and warp
    geo/ convert/ util/
  render/          render_integral, ProjectionScene, render_field_integral
  embedding/       DINOv3 extraction and the field reducers
  io/              N-channel field storage
```

### Camera (`core/rendering/camera.py`)

```python
Camera(fovy=60.0,                  # vertical field of view, degrees (perspective)
       aspect_ratio=1.0,
       orthogonal=False,
       orthogonal_size=(16, 16),   # full width/height in world units
       near=0.1, far=10000,
       position=Vector3(...), rotation=Quaternion(...))
```

### Shot (`core/rendering/shot.py`)

One captured frame plus the camera pose it was taken from, and a correction transform for
GPS/IMU drift. `CtxShot` dispatches on the context it is given, so the same call builds an
OpenGL texture, a tensor or a WebGPU texture.

### Renderer (`core/rendering/renderer.py`)

A facade that dispatches to the backend owning the context.

```python
renderer.project_shots(shots, RenderResultMode.ShotOnly, mask=mask)
renderer.render_integral(shots, mask=mask, alpha_threshold=0.1)
result = renderer.render_integral_raw(shots, mask=mask)   # unnormalised
```

`render_integral_raw` returns an `IntegralResult` carrying the accumulated samples and the
per-pixel coverage **separately**:

```python
result.accum        # (H, W, C) float32 — summed contributions
result.coverage     # (H, W)    float32 — how many shots saw each pixel
result.normalised(threshold=0.1)
```

Coverage used to be the accumulated alpha channel. Keeping it separate is what lets a light
field carry data in all four channels, and it means `alpha_threshold` thresholds an actual
overlap count rather than an opacity that happens to correlate with one.

### Coordinate conversion (`core/convert/convert.py`)

```python
world = pixel_to_world_coord(x, y, width, height, mesh, camera, include_misses=True)
xs, ys = world_to_pixel_coord(coordinates, width, height, camera)
```

---

## Rendering modes

### Orthographic

```python
settings = BaseSettings(
    orthogonal=True,
    ortho_size=(70, 70),          # world units (metres)
    camera_dist=10.0,             # height above terrain
    resolution=Resolution(2048, 2048),
)
```

Used for detection training data, mapping animal positions, and consistent-scale imagery.

### ALFS integration

```python
renderer.render_integral(shot_loader, mask=mask, alpha_threshold=2.0)
```

| Parameter | Meaning |
|---|---|
| `nr_of_frames_before_current` | temporal window, past |
| `nr_of_frames_after_current` | temporal window, future |
| `neighbor_fps` | sampling rate within the window |
| `alpha_threshold` | minimum number of overlapping shots for a pixel to count |
| `merge_labels_in_alfs` | aggregate labels from every contributing frame |

Used for noise reduction through multi-view integration, novel viewpoint synthesis, and
filling gaps behind occluders.

---

## Multi-channel and embedded light fields

The integral averages whatever the shots carry, and nothing about that requires colour. Give
it dense DINOv3 descriptors and the result is a novel view whose pixels are 1280-dimensional
feature vectors — useful for retrieval and similarity search over terrain rather than for
looking at.

```python
from alfspy.embedding import DinoV3Extractor, FieldReducer
from alfspy.render.field import FieldShot, render_field_integral
from alfspy.io.field import open_field, save_field

fields = DinoV3Extractor().extract_batch(frames)       # (h/16, w/16, 1280) per frame

out = open_field('field.npy', (2048, 2048, 1280))      # memmap; ~21 GB
result = render_field_integral(renderer, ctx, shots, out=out)
save_field('field.npy', out, metadata)                 # + field_meta.json sidecar

rgb = FieldReducer('pca').fit_transform(result.normalised(), mask=result.covered)
```

An OpenGL colour attachment holds four components, so a wider field is rendered in groups and
reassembled; coverage is rendered once and reused across them, because geometry does not
change between groups.

Fields upload at **their own resolution**, not the render resolution. Texture sampling is
already bilinear, so a 128×128 patch grid uploads as 128×128 and the GPU performs the
upsample — measured at a 256× reduction in per-slice upload traffic at 2048² output.

`FieldReducer` separates fitting from transforming, so one basis can colour a whole sequence.
Refitting per frame gives each frame its own basis and the sequence flickers. PCA, UMAP and
t-SNE are available; t-SNE reports that it cannot be reused rather than silently refitting,
since it has no out-of-sample transform.

> The DINOv3 weights are gated on HuggingFace. Set `$HF_TOKEN`, or point `DinoV3Extractor` at
> a local directory. Use `local_files_only=True` on machines with no outbound network.

---

## Ray casting

Ray–mesh intersection is used to project labels onto the DEM, and is selected exactly the
way the render backend is:

```python
ProjectionScene(dem, poses, correction, raycaster='warp')   # per scene
cast_ray(origins, directions, mesh, raycaster='warp')       # per call
```

```bash
export ALFS_RAYCASTER=warp                                  # per process
```

Precedence matches the engine: an explicit argument, then `$ALFS_RAYCASTER`, then the
default. `embree` is the default. Measured on a 131k-triangle DEM, the crossover is around 10⁴ rays:

| rays | embree | Warp CUDA |
|---|---|---|
| 80 (one frame's labels) | <1 ms | <1 ms |
| 64k | 33 ms | 2 ms |
| 4.2M (2048² dense) | 1.810 s | 0.085 s |

Label projection casts tens of rays per frame, so the GPU backend cannot win there; it is for
bulk work, and `AlfsPy[warp]` is the only ray-caster extra you would ever need to add.

`embree` needs no extra — it arrives with the base install via `trimesh[easy]`. Check it is
actually active with `EmbreeRayCaster.accelerated`: without `embreex`, trimesh falls back to a
pure-Python intersector that is not merely ~85× slower but returns hits in a different index
order, and says nothing about it.

---

## Label projection

2D bounding boxes are carried from source frames into the rendered output:

1. read source labels (YOLO or MOT format),
2. convert to pixel coordinates in source-image space,
3. project to world coordinates by ray–mesh intersection,
4. re-project into the output image through the virtual camera,
5. take the axis-aligned bounding box of the result,
6. write YOLO labels.

```python
world = pixel_to_world_coord(pixel_xs, pixel_ys, in_w, in_h, mesh, camera,
                             include_misses=False)
xs, ys = world_to_pixel_coord(world, out_w, out_h, single_shot_camera)
```

In ALFS mode with `merge_labels_in_alfs=True`, labels from every contributing frame are
collected and merged per track, with optional non-maximum suppression (`NMS_IOU`).

---

## Correction system

GPS and IMU measurements contain systematic errors that must be corrected for accurate
geo-referencing. Each shot can carry a position and rotation correction, either one default
for the flight or per frame-range:

```json
{
  "corrections": [
    {
      "start frame": 0,
      "end frame": 500,
      "translation": {"x": 0.5, "y": -0.3, "z": 1.2},
      "rotation": {"x": 0, "y": 0, "z": 0.02}
    }
  ],
  "default": {
    "translation": {"x": 0, "y": 0, "z": 0},
    "rotation": {"x": 0, "y": 0, "z": 0}
  }
}
```

Drone poses are `[tilt, roll, heading]` in degrees, with tilt measured from nadir and heading
clockwise from north. Use `quaternion_from_drone_pose`; composing them as
`quaternion_from_eulers(..., 'zyx')` turns the heading into a spin about the camera's own
optical axis, which is invisible at nadir and up to 128° wrong at the horizon.

---

## Usage

### Docker

```bash
docker build --tag orthorender -f Dockerfile .

docker run --rm \
  -v /path/to/input:/input -v /path/to/output:/output \
  -e INPUT_DIR=/input -e OUTPUT_DIR=/output \
  orthorender
```

The image sets `ALFS_ENGINE=torch` and ships the **CPU** torch build, because the CUDA wheel
adds roughly 2 GB. For GPU rendering, drop the `--index-url .../whl/cpu` line from the
`Dockerfile` and rebuild, or base the image on an `nvidia/cuda` runtime:

```bash
docker run --rm --ipc=host --gpus '"device=0"' \
  -v /path/to/input:/input -v /path/to/output:/output \
  -e INPUT_DIR=/input -e OUTPUT_DIR=/output -e ALFS_DEVICE=cuda \
  orthorender
```

The image installs no `libx11`, `mesa` or `xvfb`, and the entry point starts no virtual
display. The engine is set explicitly rather than left to a fallback, so a render in the
container cannot silently change backend if a GL driver ever appears.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALFS_ENGINE` | `moderngl` | Render backend: `moderngl`, `torch` or `vulkan` |
| `ALFS_DEVICE` | - | Device to render on, e.g. `cuda`, `cuda:1`, `cpu`. Used by torch; Vulkan maps `cpu` onto its software adapter; ModernGL ignores it |
| `ALFS_RAYCASTER` | `embree` | Ray caster: `embree` or `warp` |
| `HF_TOKEN` | - | HuggingFace token for the gated DINOv3 weights |
| `INPUT_DIR` | - | Path to input dataset folder |
| `OUTPUT_DIR` | - | Path to output folder |
| `SPLITS` | `train,val,test` | Comma-separated list of splits to process |
| `CAMERA_DISTANCE` | `10.0` | Camera height above terrain (metres) |
| `ORTHO_WIDTH` / `ORTHO_HEIGHT` | `70` | Orthographic frustum size (metres) |
| `INPUT_WIDTH` / `INPUT_HEIGHT` | `1024` | Input image size (pixels) |
| `RENDER_WIDTH` / `RENDER_HEIGHT` | `2048` | Output image size (pixels) |
| `INITIAL_SKIP` | `0` | Frames to skip at start |
| `ADD_BACKGROUND` | `1` | Overlay result on the DEM render |
| `FOVY` | `50.0` | Field of view for the perspective camera |
| `ASPECT_RATIO` | `1.0` | Camera aspect ratio |
| `SAVE_LABELED_IMAGES` | `0` | Save images with labels drawn on |
| `PROJECT_ORTHOGONAL` | `1` | Orthographic (1) or ALFS (0) mode |
| `ADDITIONAL_ROTATIONS` | `0` | Extra rotated views per frame |
| `ROTATION_LIMIT` | `2π` | Max rotation angle for augmentation |
| `ROTATION_SEED` | `-1` | Random seed (-1 for random) |
| `MERGE_LABELS_IN_ALFS` | `1` | Merge labels from all ALFS frames |
| `APPLY_NMS` | `0` | Apply non-maximum suppression |
| `NMS_IOU` | `0.9` | NMS IoU threshold |
| `IS_THERMAL` | `1` | Process thermal (1) or RGB (0) data |

### Input data structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── {flight_id}_{frame_id}.png
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/                                 # YOLO format, mirroring images/
├── correction_data/
│   ├── {flight_id}_dem.glb                 # Digital Elevation Model
│   ├── {flight_id}_matched_poses.json      # Camera poses
│   ├── {flight_id}_correction.json         # GPS/IMU corrections
│   ├── {flight_id}_mask_t.png              # Thermal mask
│   └── {flight_id}_mask_r.png              # RGB mask
├── export_train.json
├── export_val.json
└── export_test.json
```

---

## Migrating from 2.x

The public API is unchanged — `Renderer`, `CtxShot`, `ProjectionScene` and the pipeline
scripts keep their signatures, and existing call sites need no edits. What changed:

| Change | What to do |
|---|---|
| No render backend is a base dependency | `pip install "AlfsPy[moderngl]"` to keep 2.x behaviour |
| Default engine is ModernGL | Nothing, unless you want another: `engine=` or `$ALFS_ENGINE` |
| `trimesh` floor raised to >= 4.0 | Nothing. trimesh 3.x silently used a pure-Python ray intersector that returned hits in a **different index order** — a correctness bug, not just a slow path |
| `RenderObject` no longer exported from `alfspy.core.rendering` | Import it from the backend package. It is a mesh resident on a device, so it is backend-specific |
| Alpha in a returned integral image now means opacity | Read the overlap count from `IntegralResult.coverage` via `render_integral_raw` |
| Vulkan backend needs Python >= 3.11 | Nothing, unless you use it. Everything else runs on 3.9+ |
| `make_mgl_context` / `make_torch_context` deprecated | Use `make_context('moderngl')` / `make_context('torch')`. The old names warn and delegate |

Coming from **AlfsTorch** (`bambi-eco/alfs_pytorch`), which this release absorbs: the import
name was already `alfspy`, so swap the distribution for `AlfsPy[torch]` and set
`ALFS_ENGINE=torch`. `make_mgl_context` returns an OpenGL context again rather than a
`TorchContext`; use `make_context('torch')`.

---

## Known limitations

1. **DEM resolution.** Geo-referencing accuracy depends on it. High-resolution DEMs are
   recommended.

2. **Correction determination.** Correction factors are still established per flight by
   visually inspecting the alignment of static objects.

3. **CPU throughput of the torch backend.** A pure-tensor rasteriser is slower than a GPU
   driver: a 30-shot integral at 2048² over a 131k-triangle DEM takes roughly 7 s per output
   frame on CPU. Use `device='cuda'`, or the `moderngl`/`vulkan` backends. Benchmark with
   `python -m test.bench.bench_raster`.

4. **~~Virtual framebuffer artifacts~~ — resolved.** The ModernGL/Xvfb transparency artifacts
   are avoidable by choosing a backend that opens no GL context. A second, host-side cause of
   the same symptom was also fixed: `render_integral` divided with `np.divide(..., where=...)`
   and no `out=`, so pixels below the threshold held uninitialised heap memory rather than
   zeros — which is why the artifacts appeared "sometimes".

5. **~~Latent coordinate-conversion defects~~ — resolved.** The transposed ray rotation and
   its Euler-negating compensator, the `world_to_pixel_coord` broadcast that only accepted 1
   or 4 points, the never-working orthographic ray branch, and the non-invertible
   `change_pixel_origin`. See [`docs/MIGRATION_REVIEW.md`](docs/MIGRATION_REVIEW.md).

---

## Documentation

- [`docs/MIGRATION_REVIEW.md`](docs/MIGRATION_REVIEW.md) — the PyTorch port's design review
  and defect inventory
- [`docs/pyrr_conventions.md`](docs/pyrr_conventions.md) — matrix and vector conventions
- [`test/README.md`](test/README.md) — the test suite

## Acknowledgments

This project is funded by the Austrian Research Promotion Agency FFG (project THUMPER;
program number: 917796) and was developed as part of the BAMBI research project (program
number: 892231).
