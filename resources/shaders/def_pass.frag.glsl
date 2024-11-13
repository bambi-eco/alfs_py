#version 330

in vec4 v_out_v4_pos;

layout (location = 0) out vec4 f_out_v4_dir;
layout (location = 1) out vec4 f_out_v4_dist;

vec4 pack_f_to_rgba_v4(const in float fval)
{
    const vec4 bit_shift = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);
    const vec4 bit_mask  = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);
    vec4 res = fract(fval * bit_shift);
    res -= res.xxyz * bit_mask;
    return res;
}

void main() {
    vec3 normal = ((normalize(v_out_v4_pos.xyz)) + 1.0) * 0.5;
    f_out_v4_dir = vec4(normal.xyz, 1.0);
    f_out_v4_dist = pack_f_to_rgba_v4(length(v_out_v4_pos));
}