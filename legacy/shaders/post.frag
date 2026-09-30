#version 330
out vec4 fragColor;
uniform sampler2D uScene;
uniform vec2 uRes;
uniform float uBloom;
uniform float uGrain;
uniform float uTime;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec3 c = texture(uScene, uv).rgb;
    vec3 bl = vec3(0.0);
    float w = 0.0;
    for (int l = 2; l <= 7; l++) {
        float wl = 1.0 / float(l);
        vec3 s = textureLod(uScene, uv, float(l)).rgb;
        bl += max(s - 0.55, 0.0) * wl;
        w += wl;
    }
    c += bl / w * uBloom * 2.5;
    c = aces(c * 1.05);
    c += (hash12(gl_FragCoord.xy + fract(uTime) * 1000.0) - 0.5) * uGrain;
    fragColor = vec4(c, 1.0);
}
