"""Rasteriser throughput measurements.

Not part of the default test run -- ``testpaths`` in ``pyproject.toml`` collects ``test/``,
and this file has no ``test_`` functions, so pytest ignores it. Run it directly::

    python -m test.bench.bench_raster
    python -m test.bench.bench_raster --device cuda --resolution 2048

Use it to decide whether the pure-torch rasteriser is fast enough for a given workload
before reaching for the ``nvdiffrast`` escape hatch described in
``docs/MIGRATION_REVIEW.md`` section 8.1.
"""

import argparse
import os
import sys
import time

# Run directly, without an editable install: make `src` importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, 'src')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
from pyrr import Quaternion, Vector3

from alfspy.core.rendering import CtxShot, RenderResultMode, Renderer, Resolution, TextureData
from alfspy.core.torchgl import create_context
from test.helpers.scenes import (
    checkerboard_rgba,
    fovy_covering,
    gradient_rgba,
    height_field,
    ortho_camera_above,
)


def _time(callable_, repeats: int, device: str) -> float:
    """
    :return: The median wall-clock duration in milliseconds.
    """
    timings = []
    for _ in range(repeats):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        callable_()
        if device == 'cuda':
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))


def run(device: str, resolution: int, dem_resolution: int, shots: int, repeats: int) -> None:
    ctx = create_context(device=device)
    mesh = height_field(resolution=dem_resolution, half=25.0, amplitude=2.0)
    triangles = len(mesh.indices)

    camera = ortho_camera_above(size=(40.0, 40.0), height=30.0, far=200.0)
    renderer = Renderer(Resolution(resolution, resolution), ctx, camera,
                        mesh, TextureData(checkerboard_rgba(16, 32)))

    image = gradient_rgba(1024, 1024)
    shot_list = [
        CtxShot(ctx, image, position=Vector3([0.0, 0.0, 30.0]), rotation=Quaternion(),
                fovy=fovy_covering(15.0, 30.0), aspect_ratio=1.0)
        for _ in range(shots)
    ]

    try:
        print(f'device={device}  output={resolution}x{resolution}  '
              f'triangles={triangles}  shots={shots}')

        background = _time(renderer.render_background, repeats, device)
        print(f'  background render        : {background:8.1f} ms')

        single = _time(
            lambda: renderer.project_shots(shot_list[0], RenderResultMode.ShotOnly),
            repeats, device)
        print(f'  single shot projection   : {single:8.1f} ms')

        integral = _time(
            lambda: renderer.render_integral(shot_list, auto_contrast=False),
            repeats, device)
        print(f'  integral of {shots:2d} shots      : {integral:8.1f} ms '
              f'({integral / max(shots, 1):.1f} ms/shot)')
    finally:
        renderer.release()
        ctx.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='torch device (cpu or cuda)')
    parser.add_argument('--resolution', type=int, default=1024, help='square output resolution')
    parser.add_argument('--dem-resolution', type=int, default=128, help='DEM quads per side')
    parser.add_argument('--shots', type=int, default=8, help='shots to integrate')
    parser.add_argument('--repeats', type=int, default=3, help='timed repetitions')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')

    run(args.device, args.resolution, args.dem_resolution, args.shots, args.repeats)


if __name__ == '__main__':
    main()
