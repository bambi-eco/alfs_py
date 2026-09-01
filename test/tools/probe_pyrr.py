"""Prints the pyrr / GLSL matrix conventions the port depends on.

Run with ``python -m test.tools.probe_pyrr``. The output is what ``docs/pyrr_conventions.md``
documents; regenerate it after a pyrr upgrade to confirm nothing moved.
"""

import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, Vector4


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    print('=== 1. pyrr is a row-vector library ===')
    translation = Matrix44.from_translation(Vector3([1.0, 2.0, 3.0]))
    print('from_translation([1,2,3]):\n', np.asarray(translation))
    print('  row 3 =', np.asarray(translation)[3, :3], '(translation lives here)')

    point = Vector4([10.0, 20.0, 30.0, 1.0])
    print('  T * v   (pyrr)      =', np.asarray(translation * point))
    print('  v @ T   (row-vector)=', np.asarray(point) @ np.asarray(translation))
    print('  T @ v   (col-vector)=', np.asarray(translation) @ np.asarray(point), '<- wrong')

    print('\n=== 2. pyrr * composes in reverse of numpy @ ===')
    view = Matrix44.look_at(Vector3([0.0, 0.0, 5.0]), Vector3([0.0, 0.0, 0.0]),
                            Vector3([0.0, 1.0, 0.0]))
    proj = Matrix44.perspective_projection(60.0, 1.0, 0.1, 100.0)
    combined = proj * view
    print('  (P*V)_pyrr == V_np @ P_np :', np.allclose(np.asarray(combined),
                                                       np.asarray(view) @ np.asarray(proj)))
    print('  (P*V)_pyrr == P_np @ V_np :', np.allclose(np.asarray(combined),
                                                       np.asarray(proj) @ np.asarray(view)))

    print('\n=== 3. projection sanity ===')
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    clip = origin @ np.asarray(combined)
    print('  perspective, origin at distance 5 -> ndc.z =', clip[2] / clip[3], '(expect ~0.962)')

    ortho = Matrix44.orthogonal_projection(-5, 5, -5, 5, 0.1, 100.0)
    clip = np.array([2.5, 0.0, 0.0, 1.0]) @ (np.asarray(view) @ np.asarray(ortho))
    print('  orthographic, x=2.5 of half-width 5 -> ndc.x =', clip[0] / clip[3], '(expect 0.5)')

    print('\n=== 4. the rotation trap ===')
    quaternion = Quaternion.from_y_rotation(np.pi / 2)
    rotation = np.asarray(quaternion.matrix33)
    forward = np.array([0.0, 0.0, -1.0])
    print('  matrix33:\n', rotation)
    print('  FORWARD @ R33  =', forward @ rotation,
          '  <- camera->world; Transform.forward and pixel_to_world_coord use this')
    print('  R33 @ FORWARD  =', rotation @ forward, '  <- world->camera; the opposite')
    print('  pixel_to_world_coord used the second until it was fixed; see')
    print('  docs/MIGRATION_REVIEW.md section 9.3.')

    print('\n=== 5. TRS order ===')
    scale = Matrix44.from_scale(Vector3([2.0, 2.0, 2.0]))
    rotation44 = Matrix44.from_quaternion(quaternion)
    trs = translation * rotation44 * scale
    result = np.array([1.0, 0.0, 0.0, 1.0]) @ np.asarray(trs)
    print('  (1,0,0) through T(1,2,3)*R(y,90)*S(2) =', result[:3], '(expect [1,2,5])')

    print('\n=== conclusion ===')
    print('  GLSL   P * V * M * vec4(pos,1)')
    print('  numpy  pos_row @ M_pyrr @ V_pyrr @ P_pyrr')


if __name__ == '__main__':
    main()
