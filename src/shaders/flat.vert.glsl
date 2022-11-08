#version 330
uniform mat4 u_m4_proj;
uniform mat4 u_m4_view;
uniform mat4 u_m4_model;

layout (location = 0) in vec3 v_in_v3_pos;

void main() {
    gl_Position = u_m4_proj * u_m4_view * u_m4_model * vec4(v_in_v3_pos.xyz, 1.0);
}