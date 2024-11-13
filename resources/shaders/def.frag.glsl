#version 330
uniform sampler2D u_s2d_tex;
uniform sampler2D u_s2d_dir;
uniform sampler2D u_s2d_dist;

uniform mat4 u_m4_shot_view;
uniform mat4 u_m4_shot_proj;

in vec4 v_out_v4_pos;
in vec2 v_out_v2_uv;

layout (location = 0) out vec4 f_out_v4_color;

float unpack_f_from_rgba_v4(const in vec4 rgba_v4)
{
    const vec4 bit_shift = vec4(1.0/(256.0*256.0*256.0), 1.0/(256.0*256.0), 1.0/256.0, 1.0);
    float depth = dot(rgba_v4, bit_shift);
    return depth;
}

void main() {
    f_out_v4_color = v_out_v4_pos;
    /*
    vec4 dist_vec = texture(u_s2d_dist, v_out_v2_uv);
    float dist = unpack_f_from_rgba_v4(dist_vec);
    vec4 dir_vec = texture(u_s2d_dir, v_out_v2_uv) * 2.0 - 1.0;

    vec4 world_pos = dir_vec * dist;
    vec4 shotUV = u_m4_shot_proj * u_m4_shot_view * world_pos;
    vec4 uv = vec4(shotUV.xyz / shotUV.w / 2.0 + 0.5, 1.0);

    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        discard;
        f_out_v4_color = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        f_out_v4_color = vec4(texture(u_s2d_tex, uv.xy).rgb, 1.0);
    }
    */
}