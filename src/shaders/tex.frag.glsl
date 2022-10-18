#version 330

in vec2 uv_cord;

uniform sampler2D tex_sampler;

out vec4 f_color;

void main() {
    f_color = texture(tex_sampler, uv_cord);
}