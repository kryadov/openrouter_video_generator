#version 330
// Universal audio-reactive compositor. Everything is driven by uniforms so
// clip configs (clips/*.py) only set parameter curves per section.
out vec4 fragColor;

uniform vec2  uRes;
uniform float uTime;

uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform float uAspA;
uniform float uAspB;
uniform float uMix;        // burn transition A->B 0..1
uniform float uSeed;       // per-transition seed
uniform float uTransition; // 0 = paper burn, 1 = crossfade/dissolve, 2 = hard cut w/ flash

// audio
uniform float uKick;
uniform float uBass;
uniform float uMid;
uniform float uHigh;
uniform float uEnergy;
uniform float uBeatPhase;
uniform float uCumKick;

// camera
uniform float uZoom;
uniform vec2  uPan;
uniform float uRot;
uniform float uKickZoom;

// fx
uniform float uKaleido;
uniform float uKSeg;
uniform float uSpin;
uniform float uWarp;
uniform float uChroma;
uniform float uSparks;
uniform float uFlames;
uniform float uPaper;
uniform float uFold;
uniform float uNeon;
uniform float uSat;
uniform float uContrast;
uniform float uBright;
uniform vec3  uTint;
uniform float uFlash;
uniform float uVignette;
uniform float uTunnel;
uniform float uGlitch;
uniform float uLanterns;
uniform vec3  uSparkColor;

#define PI 3.14159265

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < 5; i++) { v += a * noise(p); p = m * p; a *= 0.5; }
    return v;
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

vec3 hueShift(vec3 c, float a) {
    const vec3 k = vec3(0.57735);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

vec2 mirrorUV(vec2 uv) { return 1.0 - abs(mod(uv, 2.0) - 1.0); }

// p: centered, y in [-0.5,0.5]; returns cover-fit texture uv
vec2 coverUV(vec2 p, float asp) {
    float A = uRes.x / uRes.y;
    float h = max(1.0, A / asp);
    return mirrorUV(vec2(p.x / (asp * h), p.y / h) + 0.5);
}

vec3 sampleImg(sampler2D t, float asp, vec2 p) {
    vec2 d = normalize(p + 1e-5) * uChroma * (0.004 + 0.012 * uKick);
    vec3 c;
    c.r = texture(t, coverUV(p + d, asp)).r;
    c.g = texture(t, coverUV(p, asp)).g;
    c.b = texture(t, coverUV(p - d, asp)).b;
    return c;
}

vec2 kaleido(vec2 p) {
    float r = length(p);
    float a = atan(p.y, p.x) + uCumKick * 0.15;
    float seg = 2.0 * PI / max(uKSeg, 2.0);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    return r * vec2(cos(a), sin(a));
}

vec2 camera(vec2 p) {
    // twist: "turn the wheel till the walls start to spin"
    float r = length(p);
    p = rot(uRot + uSpin * r * 2.0) * p;
    // tunnel: log-polar zoom
    if (uTunnel > 0.001) {
        float a = atan(p.y, p.x);
        vec2 tp = vec2(a / PI * 0.5, 0.25 / max(r, 0.02) + uTime * 0.25 + uCumKick * 0.05);
        p = mix(p, (fract(tp) - 0.5) * vec2(1.0, 1.0), uTunnel);
    }
    float z = uZoom * (1.0 + uKickZoom * uKick);
    p = p / z + uPan;
    // organic displacement
    if (uWarp > 0.001) {
        vec2 q = p * 2.5 + uTime * 0.05;
        p += uWarp * (vec2(fbm(q + 3.1), fbm(q - 7.7)) - 0.5) * (0.6 + 0.8 * uBass);
    }
    return p;
}

vec3 fireRamp(float x) {
    x = clamp(x, 0.0, 1.0);
    return vec3(1.5, 0.42, 0.08) * smoothstep(0.0, 0.5, x)
         + vec3(0.6, 0.5, 0.15) * smoothstep(0.55, 1.0, x);
}

float glitchShift(vec2 uv) {
    float band = floor(uv.y * 24.0 + floor(uTime * 12.0) * 3.0);
    float h = hash12(vec2(band, floor(uTime * 12.0)));
    return (h > 0.85 ? (h - 0.85) * 2.0 : 0.0) * uGlitch * (0.3 + uKick);
}

void main() {
    vec2 frag = gl_FragCoord.xy;
    vec2 suv = frag / uRes;
    vec2 p = (frag - 0.5 * uRes) / uRes.y;
    p.x += glitchShift(suv) * 0.15;

    vec2 pc = camera(p);
    vec2 pk = camera(kaleido(p) * 1.2);

    // ---- transition mask (screen-space, so it reads as paper burning) ----
    float m = 0.0;     // 1 = show B
    float ember = 0.0; // glowing edge
    float charA = 0.0;
    if (uMix > 0.0001) {
        if (uTransition < 0.5) {
            float n = fbm(p * 2.2 + uSeed * 17.0) * 0.75 + 0.25 * noise(p * 12.0 + uSeed);
            // burn grows from a random origin
            vec2 o = (hash22(vec2(uSeed, 3.7)) - 0.5) * vec2(uRes.x / uRes.y, 1.0) * 0.8;
            n += length(p - o) * 0.45;
            float front = uMix * 1.35 - 0.05;
            float d = n - front;
            m = smoothstep(0.004, -0.004, d);
            ember = smoothstep(0.035, 0.0, abs(d)) * (1.0 - smoothstep(0.98, 1.0, uMix));
            charA = smoothstep(0.14, 0.0, d) * (1.0 - m);
        } else if (uTransition < 1.5) {
            float n = fbm(p * 3.0 + uSeed * 11.0);
            m = smoothstep(n - 0.1, n + 0.1, uMix * 1.2 - 0.1);
        } else {
            m = step(0.5, uMix);
        }
    }

    vec3 A  = sampleImg(uTexA, uAspA, pc);
    vec3 B  = sampleImg(uTexB, uAspB, pc);
    if (uKaleido > 0.001) {
        vec3 Ak = sampleImg(uTexA, uAspA, pk);
        vec3 Bk = sampleImg(uTexB, uAspB, pk);
        A = mix(A, Ak, uKaleido);
        B = mix(B, Bk, uKaleido);
    }
    A *= mix(1.0, 0.08, charA);             // charring before the burn front
    A = mix(A, A * vec3(0.55, 0.3, 0.15), charA * 0.6);
    vec3 col = mix(A, B, m);
    // ---- paper: fibres + folds ----
    if (uPaper > 0.001) {
        float fib = fbm(frag * 0.012) * 0.6 + noise(frag * vec2(0.9, 0.05)) * 0.2 + noise(frag * 0.35) * 0.2;
        col *= mix(1.0, 0.82 + 0.3 * fib, uPaper);
    }
    if (uFold > 0.001) {
        vec2 fp = rot(0.35) * p * 2.0;
        vec2 cell = fract(fp) - 0.5;
        float light = (sign(cell.x) * 0.5 + sign(cell.y) * 0.3) * 0.12;
        float crease = smoothstep(0.012, 0.0, min(abs(cell.x), abs(cell.y)));
        col *= 1.0 + uFold * (light - crease * 0.25);
    }

    // ---- grade ----
    col = hueShift(col, uNeon * (1.8 + 0.6 * sin(uTime * 0.3)) );
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, uSat + uNeon * 0.5);
    col = (col - 0.5) * uContrast + 0.5;
    col *= uTint * uBright;

    col += ember * fireRamp(0.7 + 0.3 * noise(p * 40.0 + uTime * 8.0)) * 2.2;
    // ---- floating paper lanterns (soft glowing orbs drifting up) ----
    if (uLanterns > 0.001) {
        for (int i = 0; i < 2; i++) {
            float sc = 3.0 + float(i) * 2.5;
            vec2 q = p * sc + vec2(0.0, -uTime * (0.08 + 0.05 * float(i)));
            vec2 id = floor(q);
            vec2 f = fract(q) - 0.5;
            float h = hash12(id + float(i) * 13.0);
            if (h < 0.35) {
                vec2 off = (hash22(id) - 0.5) * 0.5;
                off.x += 0.08 * sin(uTime * 0.7 + h * 30.0);
                vec2 d = (f - off) * vec2(1.0, 0.8);
                float r = length(d);
                float body = smoothstep(0.16, 0.12, r);
                float glow = exp(-r * 9.0);
                float flick = 0.85 + 0.15 * sin(uTime * 9.0 + h * 50.0) + 0.4 * uKick;
                vec3 lc = mix(vec3(1.0, 0.35, 0.12), vec3(1.0, 0.75, 0.35), h * 2.5);
                col = mix(col, lc * 1.4 * flick, body * uLanterns * 0.85);
                col += lc * glow * 0.6 * uLanterns * flick;
            }
        }
    }

    // ---- flames from the bottom ----
    if (uFlames > 0.001) {
        float y = suv.y;
        float h = uFlames * (0.14 + 0.2 * uEnergy + 0.14 * uKick);
        vec2 fq = vec2(p.x * 3.0, y * 3.5 - uTime * 2.2);
        float f = fbm(fq) * 0.8 + fbm(fq * 2.3 + 5.0) * 0.4;
        float shape = (h - y) / max(h, 1e-3) + (f - 0.6) * 1.2;
        float fi = clamp(shape, 0.0, 1.0);
        col += fireRamp(fi * 0.85) * fi * 0.9;
        col *= 1.0 - 0.3 * smoothstep(0.0, 0.3, fi) * (1.0 - fi); // smoke darkening
    }

    // ---- sparks / embers rising ----
    if (uSparks > 0.001) {
        float dens = uSparks * (0.35 + 0.65 * max(uHigh, uKick));
        for (int i = 0; i < 3; i++) {
            float sc = 9.0 + float(i) * 9.0;
            float sp = 0.35 + 0.25 * float(i);
            vec2 q = p * sc;
            q.y -= uTime * sp * sc * 0.12;
            q.x += sin(q.y * 0.3 + float(i)) * 0.4;
            vec2 id = floor(q);
            vec2 f = fract(q) - 0.5;
            float h = hash12(id + float(i) * 7.3);
            if (h < dens * 0.5) {
                vec2 off = (hash22(id * 1.7) - 0.5) * 0.7;
                vec2 d = f - off;
                d.y *= 0.35; // streak
                float r = length(d);
                float tw = 0.6 + 0.4 * sin(uTime * 20.0 + h * 90.0);
                float s = smoothstep(0.06, 0.0, r) * tw + exp(-r * 30.0) * 0.5;
                col += uSparkColor * s * (1.2 + 1.5 * uKick) / (1.0 + float(i) * 0.6);
            }
        }
    }

    col += uFlash * vec3(1.0, 0.85, 0.7);
    float vig = smoothstep(1.2, 0.3, length(p * vec2(0.9, 1.1)));
    col *= mix(1.0, vig, uVignette);

    fragColor = vec4(max(col, 0.0), 1.0);
}
