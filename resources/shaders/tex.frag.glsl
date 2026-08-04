#version 330

in vec2 v_out_v2_uv;

uniform sampler2D u_s2d_tex;

out vec4 f_out_v4_color;

void main() {
    vec3 normal = normalize(texture(u_s2d_tex, v_out_v2_uv).xyz);
    f_out_v4_color = vec4(normal.xyz, 1.0);
}