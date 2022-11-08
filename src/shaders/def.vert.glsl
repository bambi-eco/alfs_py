#version 330

layout (location = 0) in vec3 v_in_v3_pos;
layout (location = 1) in vec2 v_in_v2_uv;

out vec4 v_out_v4_pos;
out vec2 v_out_v2_uv;

void main() {
    v_out_v2_uv = v_in_v2_uv;
    v_out_v4_pos = vec4(v_in_v3_pos.xyz, 1.0);
    gl_Position = v_out_v4_pos;
}