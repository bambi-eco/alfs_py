"""Renders a scene with the reference ModernGL implementation.

Run as a **subprocess** by ``test_reference_parity.py``, using the reference project's own
interpreter. That is deliberate: both projects install a package called ``alfspy``, so they
cannot coexist in one process, and each has its own dependency set (the reference needs
ModernGL and an older numpy; the port needs torch).

Usage::

    <alfs_py python> -m test.parity._reference_render <spec.json>

The spec is a JSON object with keys ``reference_src``, ``inputs`` (paths to ``.npy`` files)
and ``output``. Everything is written to and read from disk, so nothing has to be importable
on both sides.
"""

import json
import sys


def main(spec_path: str) -> int:
    with open(spec_path, 'r') as handle:
        spec = json.load(handle)

    sys.path.insert(0, spec['reference_src'])

    import numpy as np
    import moderngl as mgl
    from pyrr import Quaternion, Vector3

    from alfspy.core.rendering import (
        Camera,
        CtxShot,
        MeshData,
        RenderResultMode,
        Renderer,
        Resolution,
        TextureData,
    )

    vertices = np.load(spec['inputs']['vertices'])
    indices = np.load(spec['inputs']['indices'])
    uvs = np.load(spec['inputs']['uvs'])
    texture = np.load(spec['inputs']['texture'])
    shot_image = np.load(spec['inputs']['shot_image'])

    resolution = int(spec['resolution'])
    ortho = float(spec['ortho_size'])
    cam_height = float(spec['camera_height'])
    shot_fovy = float(spec['shot_fovy'])
    mode = spec['mode']

    ctx = mgl.create_context(standalone=True)
    ctx.enable(mgl.DEPTH_TEST)
    ctx.enable(mgl.CULL_FACE)
    ctx.cull_face = 'back'

    camera = Camera(orthogonal=True, orthogonal_size=(ortho, ortho),
                    position=Vector3([0.0, 0.0, cam_height]), near=0.1, far=500.0)

    mesh = MeshData(vertices=vertices, indices=indices, uvs=uvs)
    renderer = Renderer(Resolution(resolution, resolution), ctx, camera, mesh,
                        TextureData(texture))

    try:
        if mode == 'background':
            result = renderer.render_background()
        elif mode == 'shot':
            shot = CtxShot(ctx, shot_image, position=Vector3([0.0, 0.0, cam_height]),
                           rotation=Quaternion(), fovy=shot_fovy, aspect_ratio=1.0)
            result = renderer.project_shots(shot, RenderResultMode.ShotOnly)[0]
        elif mode == 'integral':
            shots = [
                CtxShot(ctx, shot_image,
                        position=Vector3([offset, 0.0, cam_height]), rotation=Quaternion(),
                        fovy=shot_fovy, aspect_ratio=1.0)
                for offset in spec['shot_offsets']
            ]
            result = renderer.render_integral(shots, auto_contrast=False, alpha_threshold=0.1)
        else:
            raise ValueError(f'Unknown mode "{mode}"')

        np.save(spec['output'], result)
    finally:
        renderer.release()
        ctx.release()

    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
