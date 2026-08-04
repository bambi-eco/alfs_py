# pyrr / ModernGL / GLSL matrix conventions (verified)

Everything in this port depends on getting these three conventions to line up. They were
established empirically against `pyrr==0.10.3`, not assumed. Reproduce with
`python -m test.tools.probe_pyrr`.

## 1. pyrr is a row-vector library

`Matrix44.from_translation([1,2,3])` places the translation in **row 3**:

```
[[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [1 2 3 1]]
```

Therefore a point is transformed as `v_row @ M` — **not** `M @ v_col`:

| expression | result for `v = (10,20,30,1)` |
|---|---|
| `T * v` (pyrr operator) | `(11, 22, 33, 1)` ✅ |
| `v @ T` (numpy, row-vector) | `(11, 22, 33, 1)` ✅ |
| `T @ v` (numpy, column-vector) | `(10, 20, 30, 141)` ❌ |

## 2. pyrr's `*` composes in reverse of numpy `@`

```
(P * V)_pyrr  ==  V_np @ P_np        # verified True
(P * V)_pyrr  !=  P_np @ V_np        # verified False
```

So `Matrix44.__mul__` reads left-to-right as "apply the right one first", matching the
familiar GL reading order while storing transposed data.

## 3. ModernGL's raw upload transposes into GLSL

`Program[name].write(matrix)` writes the array's raw bytes. GLSL reads a `mat4` uniform
**column-major**. A row-major pyrr array written raw is therefore seen by GLSL as its
transpose — which is exactly what turns the row-vector matrix into a column-vector one.

Consequence, and the single most important identity for this port:

```
GLSL:   P * V * M * vec4(pos, 1)
numpy:  pos_row @ M_pyrr @ V_pyrr @ P_pyrr
```

and for the shot program:

```
GLSL:   P_shot * C_shot * V_shot * world_pos
numpy:  world_pos_row @ V_shot @ C_shot @ P_shot
```

`torchgl.raster` consumes matrices in this **pyrr/row-vector form** and multiplies
left-to-right; `as_matrix()` in `torchgl/compat.py` is the single conversion point.

## 4. Verified projection sanity checks

| input | matrix | expected | got |
|---|---|---|---|
| origin, eye at `z=+5` | `perspective(60,1,0.1,100) * look_at` | `ndc.z ≈ 0.962` | `0.962` ✅ |
| `(2.5, 0, 0)` | `orthogonal(-5,5,-5,5,...)` | `ndc.x = 0.5` | `0.5` ✅ |

## 5. The rotation trap

`Quaternion.from_y_rotation(π/2)` → `matrix33 = [[0,0,1],[0,1,0],[-1,0,0]]`.

```
(0,0,-1) @ R33   = ( 1, 0, 0)      # camera -> world. pyrr's own operator agrees: R * FORWARD
R33 @ (0,0,-1)   = (-1, 0, 0)      # the other convention: world -> camera
```

`Transform.forward` uses the first, and so does everything else in the codebase.

**Historical note.** `convert.pixel_to_world_coord` used to use the *second*
(`np.dot(dirs, R33.T)`), so its centre ray pointed opposite to where the camera faced. It
was cancelled downstream by `ProjectionScene._camera_for_frame` negating its Euler angles —
a cancellation that only worked for single-axis rotations, since `R(-e) == R(e).T` fails for
composed Euler angles. Both halves were fixed together; see §9.3 of `MIGRATION_REVIEW.md`.
If you are reading old code or an old branch, this is what you are looking at.

## 6. TRS order

`Transform.mat() = trans_mat * rot_mat * scale_mat` applies **scale, then rotation, then
translation** — the standard order. Verified: `(1,0,0)` through `T(1,2,3)·R(y,90°)·S(2)`
gives `(1,2,5)`.
