"""WGSL translations of the two ALFS programs.

Ported from the authoritative GLSL in
:mod:`alfspy.core.backends.moderngl_.renderer`, cross-checked against the torch
translation in :mod:`alfspy.core.torchgl.programs`. (``resources/shaders/*.glsl`` is dead
code in this repository -- a stale earlier revision that nothing loads -- and is not the
source of truth.)

Three things differ between OpenGL and WebGPU and every one of them is silent if you get it
wrong, so each is handled explicitly and named here:

**Clip-space depth.** OpenGL clips ``z`` to ``[-w, w]``; WebGPU clips to ``[0, w]``. The
projection matrices come from pyrr and are OpenGL-style, so geometry in the near half of the
frustum would be clipped away. ``clip.z = (clip.z + clip.w) * 0.5`` remaps it.

**Framebuffer origin.** OpenGL's row 0 is the bottom of the image, WebGPU's is the top, so
the GL backend flips on readback and this one does not -- see
``wgpu_.framebuffer.img_from_fbo``.

**Texture origin.** OpenGL samples ``v = 0`` at the bottom of a texture, WebGPU at the top.
Textures are uploaded bottom-up here exactly as ``TextureData.to_bytes`` does for OpenGL, and
that upload *is* the compensation: sampling ``v`` directly then lands on the same texel. An
extra ``1 - v`` in the shader would flip it back, which is a mistake that survives casual
inspection because it leaves the image mean and the coverage unchanged and only rearranges
pixels. Verified: with these conventions the backend reproduces the ModernGL golden fixtures
bit for bit.
"""

from typing import Final

__all__ = ['OBJECT_SHADER', 'SHOT_SHADER']

# Matrices are uploaded as raw pyrr bytes, which are row-major. WGSL reads a mat4x4 as
# column-major, so the uniform is effectively the transpose -- which is exactly what makes
# the column-vector product `proj * view * model * v` here equal the row-vector product
# `v @ model @ view @ proj` the rest of the codebase uses. Do not "fix" this by transposing
# on upload without also reversing the products.
_COMMON: Final[str] = """
struct Camera {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
};

// OpenGL clips z to [-w, w]; WebGPU clips to [0, w]. The projections are OpenGL-style.
fn to_wgpu_depth(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.x, clip.y, (clip.z + clip.w) * 0.5, clip.w);
}
"""

OBJECT_SHADER: Final[str] = _COMMON + """
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var obj_texture: texture_2d<f32>;
@group(0) @binding(2) var obj_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FragmentOut {
    @location(0) colour: vec4<f32>,
    @location(1) coverage: f32,
};

@vertex
fn vs_main(@location(0) in_position: vec3<f32>,
           @location(1) in_uv: vec2<f32>) -> VertexOut {
    var out: VertexOut;
    let clip = camera.proj * camera.view * camera.model * vec4<f32>(in_position, 1.0);
    out.position = to_wgpu_depth(clip);
    out.uv = in_uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var out: FragmentOut;
    out.colour = textureSample(obj_texture, obj_sampler, in.uv);
    // Both attachments must be written; an unwritten one holds undefined values.
    out.coverage = 1.0;
    return out;
}
"""

SHOT_SHADER: Final[str] = _COMMON + """
struct Shot {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
    correction: mat4x4<f32>,
    // A vec4 rather than `f32` + `vec3` padding: WGSL aligns a vec3 to 16 bytes, which would
    // push the struct to 224 and not the 208 the buffer is sized for. Only `.x` is used.
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var shot_texture: texture_2d<f32>;
@group(0) @binding(2) var shot_sampler: sampler;
@group(0) @binding(3) var<uniform> shot: Shot;
@group(0) @binding(4) var mask_texture: texture_2d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) shot_uv: vec4<f32>,
};

struct FragmentOut {
    @location(0) colour: vec4<f32>,
    @location(1) coverage: f32,
};

@vertex
fn vs_main(@location(0) in_position: vec3<f32>) -> VertexOut {
    var out: VertexOut;
    let world = camera.model * vec4<f32>(in_position, 1.0);
    out.position = to_wgpu_depth(camera.proj * camera.view * world);
    out.shot_uv = shot.proj * shot.correction * shot.view * world;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var out: FragmentOut;
    out.colour = vec4<f32>(0.0);
    out.coverage = 0.0;

    let raw = in.shot_uv;
    // Perspective divide, then OpenGL NDC [-1, 1] into [0, 1].
    let uv = raw.xyz / raw.w / 2.0 + vec3<f32>(0.5);

    if (raw.w <= 0.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        discard;
    }

    let sample_uv = uv.xy;
    var weight = 1.0;
    if (shot.flags.x > 0.0) {
        weight = textureSample(mask_texture, shot_sampler, sample_uv).r;
    }

    out.colour = textureSample(shot_texture, shot_sampler, sample_uv) * weight;
    // Coverage carries the same weight as the samples, so a masked-out fragment contributes
    // to neither and the average stays consistent.
    out.coverage = weight;
    return out;
}
"""
