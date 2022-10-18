#version 330

layout (location = 0) in vec3 pos_in;
layout (location = 1) in vec2 uv_cord_in;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 uv_cord;

void main() {
    uv_cord = uv_cord_in.xy;
    gl_Position = projection * view * model * vec4(pos_in.xyz, 1.0);
}