"""Regenerates the golden fixtures.

Run as ``python -m test.golden.capture`` from the repository root. This overwrites the
``.npy`` files in this directory, so only run it when a change to the rendered output is
intended -- and review the resulting diff, because these fixtures are what protects the
refactor from silent drift.
"""

import os

import numpy as np

from alfspy.render.render import make_mgl_context
from test.golden.cases import CASES, render_case

_HERE = os.path.dirname(os.path.abspath(__file__))


def golden_path(name: str) -> str:
    """
    :param name: The case name.
    :return: Absolute path of the fixture for that case.
    """
    return os.path.join(_HERE, f'{name}.npy')


def main() -> None:
    ctx = make_mgl_context()
    try:
        print(f'renderer: {ctx.info["GL_RENDERER"]}')
        for name in CASES:
            img = render_case(name, ctx)
            path = golden_path(name)
            np.save(path, img)
            print(f'  {name:<18} {img.shape} {img.dtype}  '
                  f'min={img.min()} max={img.max()} mean={img.mean():.2f}  -> {os.path.basename(path)}')
    finally:
        ctx.release()


if __name__ == '__main__':
    main()
