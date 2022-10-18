#version 330


uniform sampler2D shotTexture;

in vec4 wpos;
in vec4 shotUV;
out vec4 color;

void main() {
    vec4 uv = shotUV;

    // perspective division and conversion to [0,1] from NDC
    uv = vec4(uv.xyz / uv.w / 2.0 + .5, 1.0);

    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        discard; // throw away the fragment
        color = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        // DEBUG: color = vec4(1.0, 1.0, 0.0, 1.0);
        color = vec4(texture(shotTexture, uv.xy).rgb, 1.0);
    }
}