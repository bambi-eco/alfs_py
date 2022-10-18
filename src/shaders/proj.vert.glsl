#version 330

// model view projection matrices of the focus surface (virtual camera)
uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;

// view and camera/projection matrix for one shot:
uniform mat4 m_shot_cam;
uniform mat4 m_shot_proj;

in vec3 in_position;
out vec4 wpos;
out vec4 shotUV;

void main() {
    wpos = m_model * vec4(in_position, 1.0);
    gl_Position = m_proj * m_cam * wpos;

    shotUV = m_shot_proj * m_shot_cam * wpos;
}