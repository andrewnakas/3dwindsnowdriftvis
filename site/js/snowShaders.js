// GLSL for the snow systems (WebGL2 / GLSL ES 3.00).
//
// Three GPU subsystems share the wind sampling + terrain physics chunks from
// shaders.js (inherited from the parent wind viewer):
//
//   1. Snowflake particles — spawn where it is snowing (rate-proportional),
//      advect with the 3D wind, fall at a terminal velocity set by flake
//      type, and land on the terrain.
//   2. Blowing-snow streaks — surface-hugging particles that only exist where
//      snow lies on the ground AND the 10 m wind beats the saltation
//      threshold. The signature visual of an active drift event.
//   3. Snow cover — a persistent full-domain accumulation texture:
//      R = drifted snow depth (m), G = mean pack density (kg/m3 / 1000),
//      B = hours since last snowfall (/200), A = last-step drift flux.
//      Each sim step accumulates model snowfall (density from 2 m
//      temperature), then redistributes it downwind with a mass-conserving
//      donor-cell pass whose transport capacity collapses in Winstral-
//      sheltered lee zones and concave terrain — deposition emerges where
//      capacity decreases downwind, which is SnowTran-3D's logic.

import { COMMON, TERRAIN_PHYSICS, MAX_STACK } from "./shaders.js";

// Decoding helpers for the two snow scalar tiles in the frame atlas.
// Included AFTER COMMON (uses u_clampMin/u_clampMax/u_tileScale/u_frameA/B).
export const SNOW_COMMON = `
uniform vec2 u_snowOffA;     // atlas UV offset of snow tile A
uniform vec2 u_snowOffB;
uniform vec3 u_snowAMin;     // snowHr (mm/h SWE), cumSwe (mm), t2m (degC)
uniform vec3 u_snowARange;
uniform vec3 u_snowBMin;     // depth (m), pfrozen (0-1), precip (mm/h)
uniform vec3 u_snowBRange;
uniform float u_densityScale;

// .rgb = decoded values, .a = valid (inside the HRRR domain)
vec4 snowA(vec2 pos) {
  vec2 tl = clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 t = mix(texture(u_frameA, u_snowOffA + tl), texture(u_frameB, u_snowOffA + tl), u_frameMix);
  return vec4(u_snowAMin + t.rgb * u_snowARange, step(0.5, t.a));
}
vec4 snowB(vec2 pos) {
  vec2 tl = clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 t = mix(texture(u_frameA, u_snowOffB + tl), texture(u_frameB, u_snowOffB + tl), u_frameMix);
  return vec4(u_snowBMin + t.rgb * u_snowBRange, step(0.5, t.a));
}

// Fresh-snow density from 2 m temperature (LaChapelle / Hedstrom-Pomeroy):
// ~50 kg/m3 in deep cold, wetter and denser toward the melting point.
float freshDensity(float t2mC) {
  float tc = clamp(t2mC, -40.0, 5.0);
  return clamp(50.0 + 1.7 * pow(max(tc + 15.0, 0.0), 1.5), 50.0, 250.0) * u_densityScale;
}`;

// ---------------------------------------------------------------------------
// Snowflake particle update. State:
//   pos: RG = xy (normalized domain)
//   aux: R = sigma (position on the snowfall AGL ladder), G = age/255,
//        B = frames since landing /255 (0 = airborne)
export const FLAKE_UPDATE_FRAG = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outAux;

uniform sampler2D u_statePos;
uniform sampler2D u_stateAux;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_dt;
uniform float u_windGain;
uniform float u_maxAge;
uniform float u_time;
uniform vec2 u_spawnMin;
uniform vec2 u_spawnMax;
uniform int u_substeps;
uniform float u_rateRef;    // snowfall rate (mm/h) at which spawns saturate
uniform float u_restFrames; // how long a landed flake rests before recycling

#define MAX_SUBSTEPS 4

${COMMON}
${TERRAIN_PHYSICS}
${SNOW_COMMON}

void main() {
  vec4 sp = texture(u_statePos, v_uv);
  vec4 sa = texture(u_stateAux, v_uv);
  vec2 npos = statePos(sp);
  float nsigma = sa.r;
  float age = sa.g * 255.0 + 1.0;
  float landed = sa.b * 255.0;

  vec4 wind = vec4(0.0, 0.0, 0.0, 1.0);
  if (landed < 0.5) {
    float sdt = u_dt / float(u_substeps);
    for (int k = 0; k < MAX_SUBSTEPS; k++) {
      if (k >= u_substeps) break;
      float terr = terrainHeight(npos);
      float heightM;
      wind = sampleWind(npos, nsigma, terr, heightM);
      wind.xy *= u_windGain;
      float lat = u_north - npos.y * u_latSpan;
      wind.xyz = applyTerrainPhysics(npos, wind.xyz, terr, heightM - terr, lat);

      // Terminal velocity: dendrites drift down at under 1 m/s; wetter flakes
      // near the melting point, and mixed-phase pellets, drop much faster.
      float t2m = snowA(npos).z;
      float pf = snowB(npos).y;
      float wet = max(smoothstep(-2.0, 1.5, t2m), 1.0 - pf);
      float vt = mix(0.8, 2.6, wet);
      wind.z -= vt;

      float dlon = wind.x * sdt / (111320.0 * max(cos(radians(lat)), 0.05));
      float dlat = wind.y * sdt / 110540.0;
      vec2 stepped = npos + vec2(dlon / u_lonSpan, -dlat / u_latSpan);

      float rungs = float(u_ladderLen - 1);
      nsigma = nsigma + (wind.z * sdt) / gapMeters(nsigma, terr) / rungs;
      // Terrain-following: rising ground pushes a flake up the ladder, so it
      // streams over a ridge instead of tunnelling into it.
      float terrNext = terrainHeight(stepped);
      float rise = terrNext - terr;
      if (rise > 0.0) nsigma -= rise / gapMeters(nsigma, terrNext) / rungs;
      npos = stepped;

      if (nsigma <= 0.0005) { nsigma = 0.0; landed = 1.0; break; }
      nsigma = min(nsigma, 1.0);
    }
  } else {
    landed += 1.0;
  }

  bool oob = npos.x < 0.0 || npos.x > 1.0 || npos.y < 0.0 || npos.y > 1.0
    || (landed < 0.5 && wind.a < 0.5);
  float lifetime = u_maxAge * (0.5 + rand(v_uv * 7.13));
  if (oob || age > lifetime || landed > u_restFrames) {
    vec2 seed = v_uv + fract(u_time);
    vec2 spawnPos = u_spawnMin + vec2(rand(seed), rand(seed.yx * 1.71)) * (u_spawnMax - u_spawnMin);
    // Spawn density proportional to local snowfall: accept with probability
    // rate/rateRef, otherwise go dormant (age 254 -> invisible) and retry
    // next frame. Converges to rate-proportional flake density in a few
    // frames with zero extra passes.
    vec4 a = snowA(spawnPos);
    float p = clamp(a.x / u_rateRef, 0.0, 1.0) * a.a;
    if (rand(seed * 2.61) < p) {
      npos = spawnPos;
      nsigma = 0.35 + 0.63 * rand(seed * 3.29);  // spawn aloft: real airtime
      age = 0.0;
      landed = 0.0;
    } else {
      npos = spawnPos;
      nsigma = 0.5;
      age = 254.0;   // dormant: > any lifetime, drawn fully transparent
      landed = 0.0;
    }
  }

  outPos = vec4(npos, 0.0, 1.0);
  outAux = vec4(nsigma, min(age, 254.0) / 255.0, landed / 255.0, 1.0);
}`;

// ---------------------------------------------------------------------------
// Blowing-snow streak update: exists only where snow lies AND the wind beats
// the saltation threshold; rides the bottom metres of the ladder.
export const STREAK_UPDATE_FRAG = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outAux;

uniform sampler2D u_statePos;
uniform sampler2D u_stateAux;
uniform sampler2D u_cover;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_dt;
uniform float u_windGain;
uniform float u_maxAge;
uniform float u_time;
uniform vec2 u_spawnMin;
uniform vec2 u_spawnMax;
uniform float u_saltThresh;  // 10 m wind speed (m/s) to start saltation

${COMMON}
${TERRAIN_PHYSICS}
${SNOW_COMMON}

void main() {
  vec4 sp = texture(u_statePos, v_uv);
  vec4 sa = texture(u_stateAux, v_uv);
  vec2 npos = statePos(sp);
  float nsigma = sa.r;
  float age = sa.g * 255.0 + 1.0;

  float terr = terrainHeight(npos);
  float heightM;
  vec4 wind = sampleWind(npos, nsigma, terr, heightM);
  wind.xy *= u_windGain * 1.2;  // saltating grains skim faster than the mean flow
  float lat = u_north - npos.y * u_latSpan;
  wind.xyz = applyTerrainPhysics(npos, wind.xyz, terr, heightM - terr, lat);

  float dlon = wind.x * u_dt / (111320.0 * max(cos(radians(lat)), 0.05));
  float dlat = wind.y * u_dt / 110540.0;
  npos += vec2(dlon / u_lonSpan, -dlat / u_latSpan);
  // Bounce within the saltation layer rather than climbing out of it.
  nsigma = clamp(nsigma + (rand(v_uv + fract(u_time)) - 0.55) * 0.15, 0.0, 1.0);

  float coverDepth = texture(u_cover, clamp(npos, 0.0, 1.0)).r;
  float spd10 = length(windAtAgl(npos, 10.0, terr).xy) * u_windGain;

  bool oob = npos.x < 0.0 || npos.x > 1.0 || npos.y < 0.0 || npos.y > 1.0 || wind.a < 0.5;
  float lifetime = u_maxAge * (0.3 + 0.7 * rand(v_uv * 7.13));
  if (oob || age > lifetime || coverDepth < 0.003 || spd10 < u_saltThresh * 0.8) {
    vec2 seed = v_uv + fract(u_time * 1.37);
    vec2 spawnPos = u_spawnMin + vec2(rand(seed), rand(seed.yx * 1.71)) * (u_spawnMax - u_spawnMin);
    float sTerr = terrainHeight(spawnPos);
    float sSpd = length(windAtAgl(spawnPos, 10.0, sTerr).xy) * u_windGain;
    float sCover = texture(u_cover, spawnPos).r;
    float p = clamp((sSpd - u_saltThresh) / 5.0, 0.0, 1.0) * step(0.003, sCover);
    if (rand(seed * 2.61) < p) {
      npos = spawnPos;
      nsigma = rand(seed * 5.3) * 0.4;
      age = 0.0;
    } else {
      npos = spawnPos;
      nsigma = 0.0;
      age = 254.0;   // dormant
    }
  }

  outPos = vec4(npos, 0.0, 1.0);
  outAux = vec4(nsigma, min(age, 254.0) / 255.0, 0.0, 1.0);
}`;

// ---------------------------------------------------------------------------
// Shared draw shader for flakes and streaks. Forked from the parent's
// DRAW_VERT: keeps the terrain occlusion raymarch and distance fade, drops
// the speed color ramp (snow is white), adds the landed fade and a
// speed-gated alpha for streaks.
export const SNOW_DRAW_VERT = `#version 300 es
precision highp float;
precision highp int;

uniform sampler2D u_statePosCurr;
uniform sampler2D u_statePosPrev;
uniform sampler2D u_stateAuxCurr;
uniform int u_stateSize;
uniform mat4 u_matrix;
uniform float u_west;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_altMerc;
uniform float u_streak;
uniform float u_maxAge;
uniform float u_windGain;
uniform float u_groundLock;  // 1 = streaks: draw at the surface
uniform float u_ladderBase;
uniform float u_surfaceLift;
uniform float u_restFrames;
uniform float u_speedAlpha;  // 1 = fade with wind below the saltation threshold
uniform float u_saltThresh;
uniform vec2 u_camPos;
uniform float u_camAlt;
uniform float u_occlude;
uniform float u_fadeNear;
uniform float u_fadeFar;

out float v_alpha;

const float PI = 3.141592653589793;

${COMMON}

void main() {
  int pid = gl_VertexID / 2;
  int end = gl_VertexID - pid * 2;
  ivec2 tc = ivec2(pid % u_stateSize, pid / u_stateSize);
  vec4 sc = texelFetch(u_statePosCurr, tc, 0);
  vec4 sp = texelFetch(u_statePosPrev, tc, 0);
  vec4 aux = texelFetch(u_stateAuxCurr, tc, 0);
  float sigma = aux.r;
  float landed = aux.b * 255.0;

  vec2 pc = statePos(sc);
  vec2 pp = statePos(sp);
  if (distance(pc, pp) > 0.02) pp = pc;
  vec2 pos = (end == 0) ? pc + (pp - pc) * u_streak : pc;

  float terr = terrainHeight(pc);
  float heightM;
  vec4 wind = sampleWind(pc, sigma, terr, heightM);

  float lon = u_west + pos.x * u_lonSpan;
  float lat = u_north - pos.y * u_latSpan;
  float mx = (lon + 180.0) / 360.0;
  float sm = clamp(sin(radians(lat)), -0.9999, 0.9999);
  float my = 0.5 - 0.25 * log((1.0 + sm) / (1.0 - sm)) / PI;

  // z carries only height above ground: MapLibre's custom-layer matrix already
  // lifts z=0 geometry onto the drawn terrain (see the parent's DRAW_VERT for
  // the full war story). Landed flakes and streaks sit at the surface lift.
  float agl = max(heightM - terr, 0.0);
  float aglDraw = mix(max(agl - u_ladderBase, u_surfaceLift), u_surfaceLift,
                      max(u_groundLock, step(0.5, landed)));
  gl_Position = u_matrix * vec4(mx, my, aglDraw * u_altMerc, 1.0);

  float rangeFade = 1.0;
  if (u_occlude > 0.5) {
    float dLon = (pc.x - u_camPos.x) * u_lonSpan * 111320.0 * cos(radians(lat));
    float dLat = (pc.y - u_camPos.y) * u_latSpan * 110540.0;
    float km = sqrt(dLon * dLon + dLat * dLat) / 1000.0;
    rangeFade = 1.0 - smoothstep(u_fadeNear, u_fadeFar, km);

    // Terrain occlusion: drop the flake if ground rises through the line of
    // sight (MapLibre's terrain depth is not in this framebuffer).
    float pz = terr + agl;
    vec2 toCam = u_camPos - pc;
    for (int i = 1; i <= 16; i++) {
      float t = float(i) / 17.0;
      t = t * t;
      if (terrainHeight(pc + toCam * t) > mix(pz, u_camAlt, t) + 30.0) {
        rangeFade = 0.0;
        break;
      }
    }
  }

  float endDim = (end == 0) ? 0.25 : 1.0;
  endDim *= rangeFade;
  float age = aux.g * 255.0;
  float fadeIn = clamp(age / 6.0, 0.0, 1.0);
  float fadeOut = 1.0 - smoothstep(0.6, 1.0, age / u_maxAge);
  // Landed flakes melt into the cover over their rest.
  float landFade = 1.0 - smoothstep(0.3, 1.0, landed / max(u_restFrames, 1.0));
  float speedFade = 1.0;
  if (u_speedAlpha > 0.5) {
    float spd = length(wind.xy) * u_windGain;
    speedFade = smoothstep(u_saltThresh * 0.8, u_saltThresh + 4.0, spd);
  }
  v_alpha = fadeIn * fadeOut * landFade * speedFade * endDim;
}`;

export const SNOW_DRAW_FRAG = `#version 300 es
precision highp float;
precision highp int;

in float v_alpha;
out vec4 outColor;

uniform vec3 u_color;
uniform float u_opacity;

void main() {
  outColor = vec4(u_color, 1.0) * (v_alpha * u_opacity);
}`;

// ---------------------------------------------------------------------------
// Snow cover simulation step. Full-screen pass over the ping-ponged cover
// state texture (v_uv IS the normalized domain position — same convention as
// particle pos, row 0 = north).
export const COVER_UPDATE_FRAG = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
out vec4 outState;

uniform sampler2D u_cover;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_dtH;        // sim step, forecast hours
uniform float u_qGain;      // drift transport gain (0 disables the drift pass)
uniform float u_meltGain;   // mm of depth lost per degC-hour above freezing
uniform float u_windGain;
uniform vec2 u_coverTexel;

${COMMON}
${TERRAIN_PHYSICS}
${SNOW_COMMON}

// Depth (m) leaving a cell this step, given its snowpack and exposure.
// Pomeroy-style: transport goes with u*^3 excess over a threshold that
// rises as the pack ages and densifies; capacity is amplified on windward
// slopes and convex ridges, and collapses in Winstral-sheltered lee zones.
// Deposition is not computed directly — it emerges where a donor cell's
// outflux exceeds this cell's own (flux convergence).
float outflux(vec2 p, vec2 dir, float spd, float depth, float dens, float ageH,
              float terr, float lat) {
  if (depth <= 0.0005 || spd <= 0.5) return 0.0;
  float ustar = 0.035 * spd;
  float ripe = clamp(ageH / 24.0, 0.0, 1.0) * 0.5
             + clamp((dens - 100.0) / 350.0, 0.0, 1.0) * 0.5;
  float ut = mix(0.18, 0.55, ripe);
  float ex = max(ustar * ustar - ut * ut, 0.0);
  if (ex <= 0.0) return 0.0;
  float qcap = ex * ustar;
  vec2 grad; float oc;
  terrainDerivs(p, terr, lat, grad, oc);
  float os = clamp(dot(dir, grad) * u_slopeScale, -0.5, 0.5);
  float sx = clamp(shelterTangent(p, dir, terr, lat), 0.0, 0.5);
  float f = clamp(1.0 + u_gammaS * 2.0 * os + u_gammaC * 2.0 * oc
                  - u_leeGain * 4.0 * sx, 0.05, 2.5);
  float cap = u_qGain * qcap * f * u_dtH;
  // Supply limit doubles as the CFL cap: at most a third of a cell's snow
  // moves per step, and it only moves one texel — unconditionally stable.
  return min(cap, 0.35 * depth);
}

void main() {
  vec2 pos = v_uv;
  vec4 st = texture(u_cover, pos);
  float depth = st.r;
  float dens = max(st.g * 1000.0, 50.0);
  float ageH = st.b * 200.0;

  float terr = terrainHeight(pos);
  float lat = u_north - pos.y * u_latSpan;
  vec4 sa = snowA(pos);
  float rate = max(sa.x, 0.0) * sa.a;
  float t2m = sa.z;

  // --- accumulation: model snowfall -> depth via fresh density ------------
  float rhoF = freshDensity(t2m);
  float dNew = rate * u_dtH / rhoF;   // mm/h SWE == kg/m2/h; /(kg/m3) -> m
  if (dNew > 0.0) {
    dens = (depth * dens + dNew * rhoF) / max(depth + dNew, 1e-6);
    depth += dNew;
  }
  ageH = rate > 0.2 ? 0.0 : min(ageH + u_dtH, 200.0);
  dens = min(dens + 3.0 * u_dtH, 500.0);  // settling

  // --- drift redistribution (donor-cell, mass-conserving) -----------------
  float flux = 0.0;
  float influx = 0.0;
  if (u_qGain > 0.0) {
    vec4 w10 = windAtAgl(pos, 10.0, terr);
    vec2 wv = w10.xy * u_windGain;
    float spd = length(wv);
    if (spd > 0.5 && w10.a > 0.5) {
      vec2 dir = wv / spd;
      flux = outflux(pos, dir, spd, depth, dens, ageH, terr, lat);
      // The donor sits one texel upwind (pos.y runs southward, so the north
      // wind component flips sign in texture space).
      vec2 donor = clamp(pos - vec2(dir.x, -dir.y) * u_coverTexel, 0.0, 1.0);
      vec4 stD = texture(u_cover, donor);
      float terrD = terrainHeight(donor);
      float latD = u_north - donor.y * u_latSpan;
      vec4 w10D = windAtAgl(donor, 10.0, terrD);
      vec2 wvD = w10D.xy * u_windGain;
      float spdD = length(wvD);
      if (spdD > 0.5 && w10D.a > 0.5) {
        influx = outflux(donor, wvD / spdD, spdD, stD.r,
                         max(stD.g * 1000.0, 50.0), stD.b * 200.0, terrD, latD);
      }
    }
  }
  depth = max(depth - flux + influx, 0.0);
  // wind-worked snow packs harder (and gets harder to move again)
  if (flux + influx > 1e-4) dens = min(dens + 25.0 * u_dtH, 500.0);

  // --- melt ---------------------------------------------------------------
  depth = max(depth - u_meltGain * max(t2m, 0.0) * u_dtH * 0.001, 0.0);
  depth = min(depth, 10.0);

  outState = vec4(depth, dens / 1000.0, ageH / 200.0,
                  clamp((flux + influx) * 40.0, 0.0, 1.0));
}`;

// ---------------------------------------------------------------------------
// Draped snow-cover mesh. A viewport-following grid generated entirely from
// gl_VertexID (no buffers); z = a small lift above the drawn terrain, which
// MapLibre's custom-layer matrix places on the 3D surface.
export const COVER_MESH_VERT = `#version 300 es
precision highp float;
precision highp int;

uniform mat4 u_matrix;
uniform float u_west;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform vec2 u_boxMin;
uniform vec2 u_boxMax;
uniform int u_grid;
uniform float u_altMerc;
uniform float u_surfaceLift;
uniform vec2 u_camPos;
uniform float u_camAlt;
uniform float u_occlude;
uniform float u_fadeNear;
uniform float u_fadeFar;

out vec2 v_pos;
out float v_fade;

const float PI = 3.141592653589793;

${COMMON}

void main() {
  int corner = gl_VertexID % 6;
  int quad = gl_VertexID / 6;
  ivec2 q = ivec2(quad % u_grid, quad / u_grid);
  vec2 c = vec2(
    (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0,
    (corner == 2 || corner == 4 || corner == 5) ? 1.0 : 0.0);
  vec2 gpos = (vec2(q) + c) / float(u_grid);
  vec2 pos = clamp(u_boxMin + gpos * (u_boxMax - u_boxMin), 0.0, 1.0);
  v_pos = pos;

  float lon = u_west + pos.x * u_lonSpan;
  float lat = u_north - pos.y * u_latSpan;
  float mx = (lon + 180.0) / 360.0;
  float sm = clamp(sin(radians(lat)), -0.9999, 0.9999);
  float my = 0.5 - 0.25 * log((1.0 + sm) / (1.0 - sm)) / PI;
  gl_Position = u_matrix * vec4(mx, my, u_surfaceLift * u_altMerc, 1.0);

  // Soft edge so the windowed mesh never ends in a hard seam.
  float fade = smoothstep(0.0, 0.05, min(gpos.x, 1.0 - gpos.x))
             * smoothstep(0.0, 0.05, min(gpos.y, 1.0 - gpos.y));

  if (u_occlude > 0.5) {
    float terr = terrainHeight(pos);
    float dLon = (pos.x - u_camPos.x) * u_lonSpan * 111320.0 * cos(radians(lat));
    float dLat = (pos.y - u_camPos.y) * u_latSpan * 110540.0;
    float km = sqrt(dLon * dLon + dLat * dLat) / 1000.0;
    fade *= 1.0 - smoothstep(u_fadeNear, u_fadeFar, km);
    vec2 toCam = u_camPos - pos;
    for (int i = 1; i <= 12; i++) {
      float t = float(i) / 13.0;
      t = t * t;
      if (terrainHeight(pos + toCam * t) > mix(terr, u_camAlt, t) + 40.0) {
        fade = 0.0;
        break;
      }
    }
  }
  v_fade = fade;
}`;

export const COVER_MESH_FRAG = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_pos;
in float v_fade;
out vec4 outColor;

uniform sampler2D u_cover;
uniform sampler2D u_depthRamp;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_mode;      // 0 = drifted sim depth, 1 = raw model depth
uniform float u_opacity;
uniform float u_depthMax;
uniform float u_dispGain;  // render-time drift detail from hi-res terrain
uniform float u_windGain;

${COMMON}
${TERRAIN_PHYSICS}
${SNOW_COMMON}

void main() {
  vec4 st = texture(u_cover, v_pos);
  float depth;
  float worked = 0.0;
  if (u_mode > 0.5) {
    vec4 sb = snowB(v_pos);
    depth = sb.x * sb.a;
  } else {
    depth = st.r;
    worked = st.a;
    // Render-time drift detail: the sim runs on coarse cells, but the
    // susceptibility of each spot — sheltered lee, gully, exposed crest —
    // is computable at the hi-res terrain texture's resolution. Displaying
    // depth x susceptibility redistributes (never invents) snow visually,
    // at ~3x the sim's spatial resolution, for free.
    if (u_dispGain > 0.0 && depth > 0.0005) {
      float terr = terrainHeight(v_pos);
      float lat = u_north - v_pos.y * u_latSpan;
      vec4 w10 = windAtAgl(v_pos, 10.0, terr);
      vec2 wv = w10.xy * u_windGain;
      if (length(wv) > 0.5 && w10.a > 0.5) {
        vec2 dir = normalize(wv);
        vec2 grad; float oc;
        terrainDerivs(v_pos, terr, lat, grad, oc);
        float os = clamp(dot(dir, grad) * u_slopeScale, -0.5, 0.5);
        float sx = clamp(shelterTangent(v_pos, dir, terr, lat), 0.0, 0.5);
        float f = 1.0 - u_dispGain * (1.1 * os + 0.9 * oc) + u_dispGain * 2.2 * sx;
        depth *= clamp(f, 0.15, 3.0);
      }
    }
  }

  float t = clamp(depth / u_depthMax, 0.0, 1.0);
  t = pow(t, 0.55);   // shallow snow gets most of the ramp's resolution
  vec4 col = texture(u_depthRamp, vec2(t, 0.5));
  vec3 rgb = col.rgb + worked * 0.10;   // faint shimmer where wind works the snow
  float a = col.a * v_fade * u_opacity;
  outColor = vec4(rgb, 1.0) * a;
}`;

export { MAX_STACK };
