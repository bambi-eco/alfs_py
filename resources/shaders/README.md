# GLSL shaders — reference only, not loaded

**Nothing in this project reads these files.** They are kept for historical reference.

Two things to know:

1. **They were already dead in `alfs_py`.** The ModernGL renderer embedded its two programs
   as f-strings inside `core/rendering/renderer.py`; it never loaded a `.glsl` file. The only
   code that did — `test_deferred_shading` in `src/alfspy/test.py` — referenced constants
   (`DEF_VERT_SHADER_PATH` and friends) that do not exist in `core/util/defs.py`, so it could
   not run. See `src/alfspy/scratch.py`.

2. **The two programs that mattered now live in Python.** Their translations are in
   `src/alfspy/core/torchgl/programs.py`, each quoting the GLSL it replaces:

   | GLSL program | PyTorch equivalent |
   |---|---|
   | textured mesh pass (`tex.*.glsl`-style) | `shade_object` |
   | projective texture mapping (`proj.*.glsl`-style) | `shot_clip_coords` + `shade_shot` |

If you are looking for how the renderer shades a fragment, read `programs.py`, not this
directory.
