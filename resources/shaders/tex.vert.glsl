#version 330

uniform mat4 u_m4_proj;
uniform mat4 u_m4_view;
uniform mat4 u_m4_model;

layout (location = 0) in vec3 v_in_v3_pos;
layout (location = 1) in vec2 v_in_v2_uv;

out vec2 v_out_v2_uv;

void main() {
    v_out_v2_uv = v_in_v2_uv.xy;
    gl_Position = u_m4_proj * u_m4_view * u_m4_model * vec4(v_in_v3_pos.xyz, 1.0);
}