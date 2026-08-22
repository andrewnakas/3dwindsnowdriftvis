// GLSL for the GPU particle system (WebGL2 / GLSL ES 3.00).
//
// One particle system spans a vertical stack of 1..MAX_STACK levels. Each
// particle carries (x, y) plus a continuous vertical coordinate sigma in
// [0,1] across the stack; the model's vertical velocity (omega, atlas B
// channel) moves particles between levels, so orographic uplift and
// convection are real 3D motion. A stack of 1 degenerates to classic
// single-level advection.
//
// State is two MRT targets (ping-ponged):
//   pos: FLOAT_STATE -> RG = xy;  byte mode -> webgl-wind packed hi/lo xy
//   aux: R = sigma, G = age/255 (float mode; byte mode uses stochastic drop)
//
// Atlas tile channels: R=u, G=v, B=omega, A=valid (in-domain AND above-ground).

export const MAX_STACK = 12;

export const QUAD_VERT = `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const COMMON = `
#ifdef FLOAT_STATE
vec2 statePos(vec4 s) { return s.rg; }
#else
vec2 statePos(vec4 s) { return s.ba + s.rg / 255.0; }
#endif

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

uniform sampler2D u_frameA;
uniform sampler2D u_frameB;
uniform float u_frameMix;
uniform int u_stackLen;
uniform vec2 u_tileOff[${MAX_STACK}];
uniform vec4 u_uvScale[${MAX_STACK}];   // uMin, uMax, vMin, vMax
uniform vec2 u_wScale[${MAX_STACK}];    // wMin, wMax (omega Pa/s)
uniform float u_wFactor[${MAX_STACK}];  // m/s per Pa/s at this level
uniform float u_height[${MAX_STACK}];   // meters ASL, or AGL where u_isAgl
uniform float u_isAgl[${MAX_STACK}];    // 1 = u_height is already above-ground
uniform float u_aglTop;                 // top of the above-ground levels (m)
uniform vec2 u_tileScale;
uniform vec2 u_clampMin;
uniform vec2 u_clampMax;

uniform vec2 u_terrOff;    // atlas UV of the terrain tile
uniform vec2 u_terrRange;  // hMin, hMax meters

// Standalone high-resolution terrain (own PNG, own texture unit). Regridding
// flattens slopes, so terrain physics reads this instead of the coarse atlas
// tile where it is available; RG = elevation, B = precomputed curvature.
uniform sampler2D u_terrHi;
uniform float u_hasTerrHi;
uniform vec2 u_terrHiRange;

float terrainHeight(vec2 pos) {
  if (u_hasTerrHi > 0.5) {
    vec4 t = texture(u_terrHi, clamp(pos, 0.0, 1.0));
    return mix(u_terrHiRange.x, u_terrHiRange.y, (t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0);
  }
  vec2 uv = u_terrOff + clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 t = texture(u_frameA, uv);  // terrain tile is identical in every frame
  return mix(u_terrRange.x, u_terrRange.y, (t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0);
}

// Height of stack level j ABOVE the local terrain.
//
// The 10 m and 80 m levels are already heights above ground — HRRR reports
// them that way — so they ride the terrain at exactly that height and must
// NOT have the surface elevation subtracted from them.
//
// Pressure levels are altitudes above sea level, so their height above ground
// is height - terrain. Where terrain rises past one it would go negative;
// rather than let the column invert, squeeze it into the shrinking space
// under the next valid level. That keeps the stack ordered and near the
// ground over high terrain without inventing a fixed ladder of heights.
float levelAgl(int j, float terr) {
  if (u_isAgl[j] > 0.5) return u_height[j];
  float agl = u_height[j] - terr;
  float floorAgl = u_aglTop + 40.0 * float(j);
  return agl > floorAgl ? agl : floorAgl;
}

// --- true height-above-ground sampling -------------------------------------
//
// HRRR only carries 10 m and 80 m as genuine above-ground winds; everything
// above that is a pressure surface, whose height above the ground depends on
// how high the ground is. Riding those surfaces directly means a particle's
// height above terrain changes as it crosses a valley, and over a summit the
// low ones are underground entirely.
//
// So particles instead ride a fixed AGL ladder — 10 m, 80 m, 150 m … — and
// the wind at each rung is interpolated from whichever model levels bracket
// that height over this particular terrain. 100 m AGL is genuinely 100 m
// above the ground everywhere, over valley floors and summits alike.
uniform float u_aglLadder[${MAX_STACK}];
uniform float u_useLadder;
uniform int u_ladderLen;   // rungs on the ladder (independent of source levels)

// Wind at a given height above ground, blended between the two model levels
// that straddle it.
vec4 windAtAgl(vec2 pos, float aglTarget, float terr) {
  vec2 tl = clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  float targetASL = terr + aglTarget;

  // Walk up the model levels to find the bracketing pair.
  int lo = 0;
  for (int k = 0; k < ${MAX_STACK}; k++) {
    if (k >= u_stackLen) break;
    float hk = u_isAgl[k] > 0.5 ? terr + u_height[k] : u_height[k];
    if (hk <= targetASL) lo = k;
  }
  int hi = min(lo + 1, u_stackLen - 1);

  float hLo = u_isAgl[lo] > 0.5 ? terr + u_height[lo] : u_height[lo];
  float hHi = u_isAgl[hi] > 0.5 ? terr + u_height[hi] : u_height[hi];
  float f = hHi > hLo + 1.0 ? clamp((targetASL - hLo) / (hHi - hLo), 0.0, 1.0) : 0.0;

  vec4 a = mix(texture(u_frameA, u_tileOff[lo] + tl), texture(u_frameB, u_tileOff[lo] + tl), u_frameMix);
  vec4 b = mix(texture(u_frameA, u_tileOff[hi] + tl), texture(u_frameB, u_tileOff[hi] + tl), u_frameMix);
  // A level that is underground here carries no usable wind; lean on the other.
  if (a.a < 0.5 && b.a >= 0.5) f = 1.0;
  else if (b.a < 0.5 && a.a >= 0.5) f = 0.0;

  vec3 wa = vec3(mix(u_uvScale[lo].x, u_uvScale[lo].y, a.r),
                 mix(u_uvScale[lo].z, u_uvScale[lo].w, a.g),
                 mix(u_wScale[lo].x, u_wScale[lo].y, a.b) * u_wFactor[lo]);
  vec3 wb = vec3(mix(u_uvScale[hi].x, u_uvScale[hi].y, b.r),
                 mix(u_uvScale[hi].z, u_uvScale[hi].w, b.g),
                 mix(u_wScale[hi].x, u_wScale[hi].y, b.b) * u_wFactor[hi]);
  return vec4(mix(wa, wb, f), max(a.a, b.a));
}

// Height above ground of ladder rung j, and the gap around it (for converting
// vertical velocity into movement along the ladder).
float ladderAgl(int j) { return u_aglLadder[clamp(j, 0, u_ladderLen - 1)]; }

float ladderGap(float sigma) {
  if (u_ladderLen < 2) return 1.0;
  float s = sigma * float(u_ladderLen - 1);
  int j = int(clamp(s, 0.0, float(u_ladderLen - 2)));
  return max(ladderAgl(j + 1) - ladderAgl(j), 5.0);
}

// Sampled wind at (pos, sigma): returns earth u, v (m/s), w (m/s), valid.
// heightM out = meters ASL, terrain-conforming per levelAgl.
vec4 sampleWind(vec2 pos, float sigma, float terr, out float heightM) {
  if (u_useLadder > 0.5) {
    // sigma indexes the AGL ladder, so a particle keeps its height above the
    // ground as it travels — 10 m stays 10 m up whether it is over the valley
    // floor or the summit.
    float sL = sigma * float(u_ladderLen - 1);
    int jL = u_ladderLen > 1 ? int(clamp(sL, 0.0, float(u_ladderLen - 2))) : 0;
    float fL = u_ladderLen > 1 ? clamp(sL - float(jL), 0.0, 1.0) : 0.0;
    float agl = mix(ladderAgl(jL), ladderAgl(jL + 1), fL);
    heightM = terr + agl;
    return windAtAgl(pos, agl, terr);
  }
  float s = sigma * float(u_stackLen - 1);
  int j = u_stackLen > 1 ? int(clamp(s, 0.0, float(u_stackLen - 2))) : 0;
  int j1 = min(j + 1, u_stackLen - 1);
  float f = u_stackLen > 1 ? clamp(s - float(j), 0.0, 1.0) : 0.0;
  vec2 tl = clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 a = mix(texture(u_frameA, u_tileOff[j] + tl), texture(u_frameB, u_tileOff[j] + tl), u_frameMix);
  vec4 b = mix(texture(u_frameA, u_tileOff[j1] + tl), texture(u_frameB, u_tileOff[j1] + tl), u_frameMix);
  // if one adjacent level is below ground (masked), snap to the valid one
  float fEff = f;
  if (a.a < 0.5 && b.a >= 0.5) fEff = 1.0;
  else if (b.a < 0.5 && a.a >= 0.5) fEff = 0.0;
  vec3 wa = vec3(mix(u_uvScale[j].x, u_uvScale[j].y, a.r),
                 mix(u_uvScale[j].z, u_uvScale[j].w, a.g),
                 mix(u_wScale[j].x, u_wScale[j].y, a.b) * u_wFactor[j]);
  vec3 wb = vec3(mix(u_uvScale[j1].x, u_uvScale[j1].y, b.r),
                 mix(u_uvScale[j1].z, u_uvScale[j1].w, b.g),
                 mix(u_wScale[j1].x, u_wScale[j1].y, b.b) * u_wFactor[j1]);
  heightM = terr + mix(levelAgl(j, terr), levelAgl(j1, terr), fEff);
  return vec4(mix(wa, wb, fEff), max(a.a, b.a));
}

float gapMeters(float sigma, float terr) {
  if (u_stackLen < 2) return 1.0;
  if (u_useLadder > 0.5) return ladderGap(sigma);
  float s = sigma * float(u_stackLen - 1);
  int j = int(clamp(s, 0.0, float(u_stackLen - 2)));
  return max(levelAgl(j + 1, terr) - levelAgl(j, terr), 30.0);
}
`;

// Terrain-driven flow physics. UPDATE_FRAG only — never include this in
// DRAW_VERT, which runs twice per particle and is the expensive pass.
//
// The model wind is a 12 km-scale field; terrain forces it in ways the coarse
// grid cannot show. Three effects, all faded out with height above ground:
//
//   w_oro = u . grad(h)          exact no-penetration boundary condition:
//                                flow crossing a slope must climb it
//   Ww = 1 + gS*Os + gC*Oc       Liston-Elder (MicroMet) speed weighting:
//                                windward slopes and ridge crests speed up,
//                                lee slopes and valleys slow down
//   theta -= 0.5*slope*sin(2d)   Ryan (1977) diverting: oblique flow turns
//                                toward alignment with the contours
//
// Os (slope in the wind direction) and Oc (curvature) are normalized to
// [-0.5, 0.5] so Ww stays in [0.5, 1.5] when gS + gC = 1.
export const TERRAIN_PHYSICS = `
uniform float u_tp;          // master gain; 0 disables the whole block
uniform vec2 u_terrTexel;    // one terrain cell in normalized pos units
uniform float u_oroDecayH;   // e-folding height (m AGL) of terrain influence
uniform float u_gammaS;      // MicroMet slope weight
uniform float u_gammaC;      // MicroMet curvature weight
uniform float u_slopeScale;  // slope (m/m) -> Omega_s
uniform float u_curvScale;   // curvature (1/m) -> Omega_c
uniform float u_curvLength;  // curvature length scale (m)
uniform float u_ryanGain;
uniform float u_leeGain;     // lee sheltering strength; 0 skips the raymarch
uniform float u_leeDistM;    // how far upwind to look for sheltering terrain

// Winstral's Sx: the steepest slope to any point upwind within u_leeDistM.
// Positive means something upwind stands higher than here, so this spot sits
// in a wake and the flow over it is slack. Speed-up and lift alone cannot
// produce that — they only know the ground directly underfoot, which is why
// terrain downscaling schemes carry this term separately.
float shelterTangent(vec2 pos, vec2 dirN, float terr, float lat) {
  float mx = 0.0;
  float mPerLon = 111320.0 * max(cos(radians(lat)), 0.05);
  for (int i = 1; i <= 8; i++) {
    float d = u_leeDistM * float(i) / 8.0;
    // step upwind; pos.y runs southward so the north component flips
    vec2 up = pos - vec2(dirN.x * d / mPerLon / u_lonSpan,
                        -dirN.y * d / 110540.0 / u_latSpan);
    mx = max(mx, (terrainHeight(up) - terr) / d);
  }
  return mx;
}

// Central-difference terrain gradient (m/m, x=east y=north) plus curvature,
// normalized to [-0.5, 0.5]. Note pos.y increases SOUTHWARD, so the north
// sample is at -y and the north-south difference is (hN - hS).
void terrainDerivs(vec2 pos, float terr, float lat, out vec2 grad, out float omegaC) {
  float hE = terrainHeight(pos + vec2(u_terrTexel.x, 0.0));
  float hW = terrainHeight(pos - vec2(u_terrTexel.x, 0.0));
  float hN = terrainHeight(pos - vec2(0.0, u_terrTexel.y));
  float hS = terrainHeight(pos + vec2(0.0, u_terrTexel.y));
  float dxm = u_lonSpan * u_terrTexel.x * 111320.0 * max(cos(radians(lat)), 0.05);
  float dym = u_latSpan * u_terrTexel.y * 110540.0;
  grad = vec2((hE - hW) / (2.0 * dxm), (hN - hS) / (2.0 * dym));

  if (u_hasTerrHi > 0.5) {
    // Precomputed: normalizing curvature needs a domain-wide maximum, which
    // only the pipeline can know. 0.5 encodes flat ground.
    omegaC = texture(u_terrHi, clamp(pos, 0.0, 1.0)).b - 0.5;
  } else {
    // positive on convex ground (ridges, knobs), negative in bowls and valleys
    float curvRaw = ((terr - 0.5 * (hW + hE)) + (terr - 0.5 * (hN + hS))) / (4.0 * u_curvLength);
    omegaC = clamp(curvRaw * u_curvScale, -0.5, 0.5);
  }
}

vec3 applyTerrainPhysics(vec2 pos, vec3 wind, float terr, float agl, float lat) {
  float fade = u_tp * exp(-max(agl, 0.0) / u_oroDecayH);
  if (fade < 0.01 || length(wind.xy) < 0.1) return wind;

  vec2 grad;
  float omegaC;
  terrainDerivs(pos, terr, lat, grad, omegaC);
  vec2 dir = normalize(wind.xy);

  float omegaS = clamp(dot(dir, grad) * u_slopeScale, -0.5, 0.5);
  float Ww = clamp(1.0 + u_gammaS * omegaS + u_gammaC * omegaC, 0.5, 1.5);

  // Ryan diverting: xi = uphill azimuth, d = angle from the wind to uphill.
  // sin(2d) vanishes for straight up/downslope and cross-slope flow, and peaks
  // at 45 degrees of obliquity — where terrain steering is strongest.
  // Turning stays small (a few degrees) because it is proportional to the
  // terrain angle, and a 2.1 km grid resolves ridges as 5-13 degree ramps
  // rather than the 30+ degree walls they are. That is the grid's limit, not
  // a weak constant — raise u_ryanGain to exaggerate, at the cost of fidelity.
  float theta = atan(wind.y, wind.x);
  float xi = atan(grad.y, grad.x);
  float d = mod(xi - theta + 3.14159265, 6.28318531) - 3.14159265;
  float delta = 0.0;
  if (abs(d) <= 1.57079633) {
    float slopeDeg = degrees(atan(length(grad)));
    delta = -0.5 * radians(min(slopeDeg, 45.0)) * sin(2.0 * d) * u_ryanGain;
  }
  delta = clamp(delta, -0.5236, 0.5236) * fade;

  float c = cos(delta), s = sin(delta);
  vec2 h = mat2(c, s, -s, c) * wind.xy;
  h *= mix(1.0, Ww, fade);

  if (u_leeGain > 0.0) {
    float sx = clamp(shelterTangent(pos, dir, terr, lat), 0.0, 0.5);
    h *= 1.0 - u_leeGain * sx * 2.0 * fade;
  }

  // lift from the deflected wind: flow steered along the contours climbs less
  float wOro = dot(h, grad) * fade;
  return vec3(h, wind.z + wOro);
}
`;

export const UPDATE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outAux;

uniform sampler2D u_statePos;
uniform sampler2D u_stateAux;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_dt;
uniform float u_windGain;  // 1 = real forecast; >1 simulates a windier day
uniform float u_maxAge;
uniform float u_time;
uniform vec2 u_spawnMin;   // respawn region (normalized), follows the viewport
uniform vec2 u_spawnMax;
uniform int u_substeps;
uniform float u_mixRate;   // 1/s relaxation back toward the parcel's level

#define MAX_SUBSTEPS 4

${COMMON}
${TERRAIN_PHYSICS}

void main() {
  vec4 sp = texture(u_statePos, v_uv);
  vec4 sa = texture(u_stateAux, v_uv);
  vec2 pos = statePos(sp);
  float sigma = sa.r;

  // Integrate in substeps when the terrain is fine enough that one 90 s jump
  // would clear a whole cell — a particle that leaps a ridge never feels it.
  vec2 npos = pos;
  float nsigma = sigma;
  vec4 wind = vec4(0.0);
  float sdt = u_dt / float(u_substeps);
  for (int k = 0; k < MAX_SUBSTEPS; k++) {
    if (k >= u_substeps) break;
    float terr = terrainHeight(npos);
    float heightM;
    wind = sampleWind(npos, nsigma, terr, heightM);
    // Scale the wind BEFORE the terrain physics, so a simulated gale drives
    // stronger lift and ridge speed-up the way a real one would — the
    // orographic terms are all proportional to wind speed.
    wind.xy *= u_windGain;
    float lat = u_north - npos.y * u_latSpan;
    wind.xyz = applyTerrainPhysics(npos, wind.xyz, terr, heightM - terr, lat);

    float dlon = wind.x * sdt / (111320.0 * max(cos(radians(lat)), 0.05));
    float dlat = wind.y * sdt / 110540.0;
    vec2 stepped = npos + vec2(dlon / u_lonSpan, -dlat / u_latSpan);

    if (u_stackLen > 1) {
      float rungs = float((u_useLadder > 0.5 ? u_ladderLen : u_stackLen) - 1);
      nsigma = clamp(nsigma + (wind.z * sdt) / gapMeters(nsigma, terr) / rungs, 0.0, 1.0);

      // Air cannot flow into rock. The orographic lift above is a tendency, not
      // a constraint, so on a slope steeper than the wind can climb a particle
      // would simply pass through the hillside. Carry the parcel over the
      // ground it actually moved onto: rising terrain pushes it up the ladder,
      // falling terrain lets it settle, which is what makes flow climb a
      // windward face and pour down the lee.
      float terrNext = terrainHeight(stepped);
      float rise = terrNext - terr;
      if (rise != 0.0) {
        nsigma = nsigma - rise / gapMeters(nsigma, terrNext) / rungs;
      }
      // Relax back toward the level this parcel belongs to. Terrain-following
      // and orographic lift are both one-way over a ridge, so without a
      // restoring term every parcel ends up pinned at the top or scraping the
      // ground and stays there. Mixing is what keeps a surface layer a layer —
      // it is also what lets a parcel climb a windward face and then settle
      // again on the lee instead of riding off into the ceiling.
#ifdef FLOAT_STATE
      float homeLevel = sa.b;
#else
      // byte mode packs position into aux/pos and has no spare channel for a
      // home level, so relax toward mid-stack instead.
      float homeLevel = 0.5;
#endif
      nsigma = mix(nsigma, homeLevel, clamp(u_mixRate * sdt, 0.0, 0.25));
      nsigma = clamp(nsigma, 0.0, 1.0);
    }
    npos = stepped;
  }

  bool oob = npos.x < 0.0 || npos.x > 1.0 || npos.y < 0.0 || npos.y > 1.0 || wind.a < 0.5;
  vec2 seed = v_uv + fract(u_time);
  vec2 spawnPos = u_spawnMin + vec2(rand(seed), rand(seed.yx * 1.71)) * (u_spawnMax - u_spawnMin);
  // bias respawn toward the surface: most flow reads at ground level, the
  // rest thins out with altitude (half of all particles in the lowest levels)
  float sr = rand(seed * 2.61);
  float spawnSigma = sr * sr * sr;

#ifdef FLOAT_STATE
  float age = sa.g * 255.0 + 1.0;
  float lifetime = u_maxAge * (0.5 + rand(v_uv * 7.13));
  float home = sa.b;
  if (oob || age > lifetime) { npos = spawnPos; nsigma = spawnSigma; age = 0.0; home = spawnSigma; }
  outPos = vec4(npos, 0.0, 1.0);
  // aux.b keeps the level this parcel belongs to, so the mixing term above
  // has something to relax toward.
  outAux = vec4(nsigma, age / 255.0, home, 1.0);
#else
  float speedN = length(wind.xy) / 60.0;
  float drop = step(1.0 - (0.006 + speedN * 0.012), rand(seed * 3.37));
  if (oob || drop > 0.5) { npos = spawnPos; nsigma = spawnSigma; }
  // byte mode has no spare channel for a home level, so it relaxes toward the
  // middle of the stack instead — enough to stop parcels pinning at one end.
  npos += (vec2(rand(seed * 5.1), rand(seed * 9.3)) - 0.5) / 65280.0;
  nsigma += (rand(seed * 6.7) - 0.5) / 255.0;  // dither vertical quantization
  outPos = vec4(fract(clamp(npos, 0.0, 1.0) * 255.0), floor(clamp(npos, 0.0, 1.0) * 255.0) / 255.0);
  outAux = vec4(clamp(nsigma, 0.0, 1.0), 0.0, 0.0, 1.0);
#endif
}`;

export const DRAW_VERT = `#version 300 es
precision highp float;

uniform sampler2D u_statePosCurr;
uniform sampler2D u_statePosPrev;
uniform sampler2D u_stateAuxCurr;
uniform int u_stateSize;
uniform mat4 u_matrix;
uniform float u_west;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_altMerc;   // mercator z per meter of height-above-terrain (altitude scale)
uniform float u_exagMerc;  // mercator z per meter of terrain (matches map terrain exaggeration)
uniform float u_streak;
uniform float u_maxAge;
uniform float u_windGain;    // matches the update pass, so colour tracks speed
uniform float u_groundLock; // 1 = draw every parcel at the surface
uniform float u_yOffset;     // manual vertical trim, metres
uniform float u_ladderBase;  // AGL height of the lowest rung, so it draws at 0
uniform float u_surfaceLift; // clearance over the terrain mesh, metres
uniform vec2 u_camPos;    // camera ground position, normalized domain coords
uniform float u_camAlt;   // camera altitude, metres ASL
uniform float u_occlude;  // enable distance fade
uniform float u_fadeNear; // km at which particles start thinning
uniform float u_fadeFar;  // km at which they are gone

out float v_speed;
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

  vec2 pc = statePos(sc);
  vec2 pp = statePos(sp);
  if (distance(pc, pp) > 0.02) pp = pc;
  vec2 pos = (end == 0) ? pc + (pp - pc) * u_streak : pc;

  float terr = terrainHeight(pc);
  float heightM;
  vec4 wind = sampleWind(pc, sigma, terr, heightM);
  v_speed = clamp(length(wind.xy) * u_windGain / 60.0, 0.0, 1.0);

  float lon = u_west + pos.x * u_lonSpan;
  float lat = u_north - pos.y * u_latSpan;
  float mx = (lon + 180.0) / 360.0;
  float sm = clamp(sin(radians(lat)), -0.9999, 0.9999);
  float my = 0.5 - 0.25 * log((1.0 + sm) / (1.0 - sm)) / PI;

  // Terrain base rises with the map's own exaggeration so particles hug the
  // rendered surface; height above ground gets the (separate) altitude scale.
  //
  // The column is anchored so its LOWEST rung sits on the ground rather than
  // starting part-way up: subtracting the base height means the 10 m wind is
  // drawn at the terrain surface, and everything above it keeps its true
  // spacing. That gives the whole vertical spectrum with nothing floating —
  // no camera angle can slip beneath the field.
  // The clearance is a FLOOR, not an offset: adding it to every rung lifted
  // the whole column and left a bare gap under the wind — a ceiling you could
  // stand beneath. As a floor, the lowest rung rests just clear of the mesh
  // and everything above keeps its true height.
  // MapLibre's matrix already lifts custom-layer geometry onto the terrain
  // when 3D terrain is enabled: a vertex at z = 0 lands exactly on the drawn
  // ground, while z = terrain lands a full terrain-height up in the sky. So z
  // carries ONLY the height above ground — adding the terrain here as well was
  // drawing every particle twice as high as the ground it belongs to, which is
  // why the trim needed to cancel it tracked the local elevation (about
  // -1500 m at Bozeman, near zero on the coast).
  // Surface mode pins every particle to the same small height above ground.
  // sigma still decides WHICH wind a parcel feels — that is the physics — but
  // it no longer decides where the parcel is DRAWN. Letting it do both meant a
  // parcel drifting up the 60 m ladder visibly rose off the hillside as it
  // climbed a ridge and sank on the lee, and clipped through ground it should
  // have been lying on. Pinned, the field is a skin on the terrain at every
  // elevation and no camera angle can get under it.
  float agl = max(heightM - terr, 0.0);
  float aglDraw = mix(max(agl - u_ladderBase, u_surfaceLift),
                      u_surfaceLift,
                      u_groundLock) + u_yOffset;
  float z = aglDraw * u_altMerc;
  gl_Position = u_matrix * vec4(mx, my, z, 1.0);

  // Fade out with distance from the camera. At a near-horizontal view the
  // horizon is hundreds of km away, and far particles — which are legitimately
  // high air — pile into the few rows of pixels above the skyline and read as
  // a slab of haze detached from the terrain. Thinning them with range keeps
  // the frame about the ground you are actually looking at.
  float rangeFade = 1.0;
  if (u_occlude > 0.5) {
    float dLon = (pc.x - u_camPos.x) * u_lonSpan * 111320.0 * cos(radians(lat));
    float dLat = (pc.y - u_camPos.y) * u_latSpan * 110540.0;
    float km = sqrt(dLon * dLon + dLat * dLat) / 1000.0;
    rangeFade = 1.0 - smoothstep(u_fadeNear, u_fadeFar, km);

    // Hide wind that terrain in front of it should be blocking. MapLibre's
    // terrain depth is not in this framebuffer, so the GPU depth test cannot
    // do it: from a low camera, ground beyond a ridge paints straight over the
    // mountain and reads as a slab hanging in the sky. Walk the terrain along
    // the line of sight and drop the particle if the ground rises through it.
    // Sample densely near the particle, where the blocking ridge usually is:
    // an evenly-spaced march steps straight over a sharp crest and lets the
    // wind behind it through.
    float pz = terr + max(heightM - terr, 0.0);
    vec2 toCam = u_camPos - pc;
    for (int i = 1; i <= 16; i++) {
      float t = float(i) / 17.0;
      t = t * t;                       // bunch the samples toward the particle
      // 30 m of slack so an undulating surface does not erase its own wind
      if (terrainHeight(pc + toCam * t) > mix(pz, u_camAlt, t) + 30.0) {
        rangeFade = 0.0;
        break;
      }
    }
  }

  float endDim = (end == 0) ? 0.2 : 1.0;
  endDim *= rangeFade;
#ifdef FLOAT_STATE
  float age = aux.g * 255.0;
  float fadeIn = clamp(age / 10.0, 0.0, 1.0);
  float fadeOut = 1.0 - smoothstep(0.6, 1.0, age / u_maxAge);
  v_alpha = fadeIn * fadeOut * endDim;
#else
  v_alpha = 0.95 * endDim;
#endif
}`;

export const DRAW_FRAG = `#version 300 es
precision highp float;

in float v_speed;
in float v_alpha;
out vec4 outColor;

uniform sampler2D u_ramp;
uniform float u_opacity;

void main() {
  vec3 color = texture(u_ramp, vec2(v_speed, 0.5)).rgb;
  outColor = vec4(color, 1.0) * (v_alpha * u_opacity);
}`;
