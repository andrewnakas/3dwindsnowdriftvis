// MapLibre custom layer orchestrating the three snow subsystems: falling
// snowflakes, blowing-snow streaks, and the persistent drifted snow cover.
// One layer (not three) so they share a single FrameManager, one copy of the
// terrain texture, and a controlled draw order: cover, then streaks, then
// flakes.

import { QUAD_VERT, MAX_STACK } from "./shaders.js";
import {
  FLAKE_UPDATE_FRAG, STREAK_UPDATE_FRAG, SNOW_DRAW_VERT, SNOW_DRAW_FRAG,
  COVER_UPDATE_FRAG, COVER_MESH_VERT, COVER_MESH_FRAG,
} from "./snowShaders.js";
import {
  depthRampData, DEPTH_MAX, SNOW_LADDER, DRIFT_LADDER, ladderSourceIndices,
} from "./atmosphere.js";
import { FrameManager } from "./frames.js";

const FLAKE_MAX_AGE = 110; // frames
const STREAK_MAX_AGE = 60;

function compile(gl, vertSrc, fragSrc, defines = "") {
  const inject = (src) => src.replace("#version 300 es", `#version 300 es\n${defines}`);
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, inject(vertSrc)], [gl.FRAGMENT_SHADER, inject(fragSrc)]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("shader: " + gl.getShaderInfoLog(sh));
    }
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("link: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

function uniforms(gl, prog) {
  const out = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(prog, i);
    out[info.name.replace("[0]", "")] = gl.getUniformLocation(prog, info.name);
  }
  return out;
}

class ParticleSystem {
  constructor(gl, size, maxAge) {
    this.size = size;
    this.gl = gl;
    // ping-pong pairs of MRT state: [ {pos, aux}, {pos, aux} ]
    this.state = [this.makeState(maxAge), this.makeState(maxAge)];
    this.cur = 0;
    this.fbo = gl.createFramebuffer();
  }

  makeState(maxAge) {
    const gl = this.gl;
    const n = this.size;
    const make = (f32) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n, n, 0, gl.RGBA, gl.FLOAT, f32);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return tex;
    };
    const count = n * n;
    const posF = new Float32Array(count * 4);
    const auxF = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      posF[i * 4] = Math.random();
      posF[i * 4 + 1] = Math.random();
      auxF[i * 4] = Math.random();
      // start everyone dormant so flakes only appear where it is snowing
      auxF[i * 4 + 1] = (Math.random() * maxAge) / 255 + 0.6;
    }
    return { pos: make(posF), aux: make(auxF) };
  }

  swap() { this.cur = 1 - this.cur; }
  get curState() { return this.state[this.cur]; }
  get prevState() { return this.state[1 - this.cur]; }

  destroy() {
    for (const s of this.state) {
      this.gl.deleteTexture(s.pos);
      this.gl.deleteTexture(s.aux);
    }
    this.gl.deleteFramebuffer(this.fbo);
  }
}

export class SnowSystem {
  constructor(map, meta, opts = {}) {
    this.id = "snow-system";
    this.type = "custom";
    this.renderingMode = "3d";
    this.map = map;
    this.meta = meta;
    this.time = 0;         // target forecast hour (UI slider)
    this.simTime = 0;      // hour the cover sim has integrated to
    this.speedFactor = 1.0;
    this.windGain = 1.0;   // scales the real wind (affects drift physics too)
    this.exaggeration = opts.exaggeration ?? 1;
    this.altScale = opts.altScale ?? this.exaggeration;

    this.flakesOn = true;
    this.streaksOn = true;
    this.coverOn = true;
    this.modelDepthMode = false; // show raw HRRR depth instead of drifted sim
    this.coverOpacity = 0.9;
    this.flakeOpacity = 0.9;
    this.qGain = 1.0;        // drift transport gain
    this.dispGain = 1.0;     // render-time drift detail gain
    this.densityScale = 1.0; // fresh-snow density multiplier
    this.meltGain = 1.0;     // mm depth per degC-hour
    this.saltThresh = 5.5;   // m/s 10 m wind to start saltation
    this.depthMax = DEPTH_MAX;
    this.depthOcclusion = opts.depthOcclusion ?? true;

    const mobile = opts.mobile ?? false;
    this.flakeCount = mobile ? 65536 : 262144;
    this.streakCount = mobile ? 16384 : 65536;
    this.coverW = mobile ? 675 : 1350;
    this.coverH = mobile ? 398 : 795;
    this.meshGrid = mobile ? 96 : 160;

    this.levelIndices = ladderSourceIndices(meta);
    this.onReady = opts.onReady;
    this.catchingUp = false;

    // Terrain-driven flow physics tuning (see TERRAIN_PHYSICS in shaders.js).
    this.terrainPhysics = opts.terrainPhysics ?? true;
    this.tuning = {
      gain: 0.7,
      oroDecayH: 800,
      gammaS: 0.5,
      gammaC: 0.5,
      slopeScale: 15.0,
      slopeScaleHi: 3.1,
      curvScale: 35.0,
      curvLength: 12800.0,
      ryanGain: 1.0,
      leeGain: 0.6,
      leeDistM: 10000.0,
    };
  }

  onAdd(map, gl) {
    if (typeof WebGL2RenderingContext === "undefined" || !(gl instanceof WebGL2RenderingContext)) {
      throw new Error("WebGL2 required");
    }
    this.gl = gl;
    this.floatOK = !!gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("OES_texture_float_linear"); // nicer cover filtering if present
    this.linearFloat = !!gl.getExtension("OES_texture_float_linear");

    const defines = "#define FLOAT_STATE";
    if (this.floatOK) {
      this.flakeProg = compile(gl, QUAD_VERT, FLAKE_UPDATE_FRAG, defines);
      this.flakeU = uniforms(gl, this.flakeProg);
      this.streakProg = compile(gl, QUAD_VERT, STREAK_UPDATE_FRAG, defines);
      this.streakU = uniforms(gl, this.streakProg);
      this.drawProg = compile(gl, SNOW_DRAW_VERT, SNOW_DRAW_FRAG, defines);
      this.drawU = uniforms(gl, this.drawProg);
      this.coverProg = compile(gl, QUAD_VERT, COVER_UPDATE_FRAG, defines);
      this.coverU = uniforms(gl, this.coverProg);
    }
    this.meshProg = compile(gl, COVER_MESH_VERT, COVER_MESH_FRAG, defines);
    this.meshU = uniforms(gl, this.meshProg);
    this.vao = gl.createVertexArray();

    // Stand-in for samplers whose real texture has not loaded yet.
    this.blankTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.blankTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 128, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.rampTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, depthRampData());
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.frames = new FrameManager(gl, this.meta);
    this.frames.loadTerrain(() => this.map.triggerRepaint());
    this.snowU = this.buildSnowUniforms();
    this.rebuildSystems();
    if (this.floatOK) this.makeCover();
    this.onReady?.(this);
  }

  onRemove() {
    this.flakes?.destroy();
    this.streaks?.destroy();
    this.destroyCover();
    this.flakes = this.streaks = null;
  }

  // --- setup helpers --------------------------------------------------------

  buildSnowUniforms() {
    const s = this.meta.snow;
    if (!s) return null;
    const { cols, rows } = this.meta.atlas;
    const off = (idx) => [(idx % cols) / cols, Math.floor(idx / cols) / rows];
    const minMax = (tile) => {
      const mins = [], ranges = [];
      for (const ch of Object.values(tile.channels)) {
        mins.push(ch.min);
        ranges.push(ch.max - ch.min);
      }
      return { mins, ranges };
    };
    const a = minMax(s.tileA), b = minMax(s.tileB);
    return {
      offA: off(s.tileA.index), offB: off(s.tileB.index),
      aMin: a.mins, aRange: a.ranges, bMin: b.mins, bRange: b.ranges,
    };
  }

  setSnowUniforms(gl, U) {
    const s = this.snowU;
    if (!s) return;
    gl.uniform2fv(U.u_snowOffA, s.offA);
    gl.uniform2fv(U.u_snowOffB, s.offB);
    gl.uniform3fv(U.u_snowAMin, s.aMin);
    gl.uniform3fv(U.u_snowARange, s.aRange);
    gl.uniform3fv(U.u_snowBMin, s.bMin);
    gl.uniform3fv(U.u_snowBRange, s.bRange);
    gl.uniform1f(U.u_densityScale, this.densityScale);
  }

  rebuildSystems() {
    if (!this.gl || !this.floatOK) return;
    this.flakes?.destroy();
    this.streaks?.destroy();
    const size = (n) => Math.max(16, 1 << Math.floor(Math.log2(Math.sqrt(n))));
    this.flakes = new ParticleSystem(this.gl, size(this.flakeCount), FLAKE_MAX_AGE);
    this.streaks = new ParticleSystem(this.gl, size(this.streakCount), STREAK_MAX_AGE);
    const levels = this.levelIndices
      .map((i) => this.meta.levels[i])
      .sort((a, b) => a.heightMeters - b.heightMeters);
    this.flakeStack = this.buildStackUniforms(levels, SNOW_LADDER);
    this.streakStack = this.buildStackUniforms(levels, DRIFT_LADDER);
  }

  setFlakeCount(n) { this.flakeCount = n; this.rebuildSystems(); }

  makeCover() {
    const gl = this.gl;
    this.destroyCover();
    const make = () => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.coverW, this.coverH, 0,
        gl.RGBA, gl.FLOAT, null);
      const filt = this.linearFloat ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return tex;
    };
    this.cover = [make(), make()];
    this.coverCur = 0;
    this.coverFbo = gl.createFramebuffer();
    this.resetSim();
  }

  destroyCover() {
    if (!this.cover) return;
    for (const t of this.cover) this.gl.deleteTexture(t);
    this.gl.deleteFramebuffer(this.coverFbo);
    this.cover = null;
  }

  resetSim() {
    const gl = this.gl;
    if (!this.cover) return;
    const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.coverFbo);
    for (const t of this.cover) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.clearColor(0, 0.1, 0, 0);   // depth 0, density 100 kg/m3, age 0
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
    this.simTime = 0;
  }

  // Read back a rectangle of the cover state (normalized domain coords) for
  // debugging and tests: returns {w, h, data} with RGBA floats per texel.
  debugReadCover(x0 = 0, y0 = 0, x1 = 1, y1 = 1) {
    const gl = this.gl;
    if (!this.cover) return null;
    const px0 = Math.floor(x0 * this.coverW), py0 = Math.floor(y0 * this.coverH);
    const w = Math.max(1, Math.floor((x1 - x0) * this.coverW));
    const h = Math.max(1, Math.floor((y1 - y0) * this.coverH));
    const out = new Float32Array(w * h * 4);
    const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.coverFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D,
      this.cover[this.coverCur], 0);
    gl.readPixels(px0, py0, w, h, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
    return { w, h, data: out };
  }

  buildStackUniforms(levels, aglLadder) {
    const { cols, rows } = this.meta.atlas;
    const L = levels.length;
    const tileOff = new Float32Array(MAX_STACK * 2);
    const uvScale = new Float32Array(MAX_STACK * 4);
    const wScale = new Float32Array(MAX_STACK * 2);
    const wFactor = new Float32Array(MAX_STACK);
    const height = new Float32Array(MAX_STACK);
    const isAgl = new Float32Array(MAX_STACK);
    for (let k = 0; k < MAX_STACK; k++) {
      const lv = levels[Math.min(k, L - 1)];
      tileOff[k * 2] = (lv.index % cols) / cols;
      tileOff[k * 2 + 1] = Math.floor(lv.index / cols) / rows;
      uvScale.set([lv.uMin, lv.uMax, lv.vMin, lv.vMax], k * 4);
      wScale.set([lv.wMin ?? -1, lv.wMax ?? 1], k * 2);
      wFactor[k] = lv.wFactor ?? 0;
      height[k] = lv.heightMeters;
      isAgl[k] = lv.kind === "height_agl" ? 1 : 0;
    }
    const aglTop = levels.reduce(
      (m, lv) => (lv.kind === "height_agl" ? Math.max(m, lv.heightMeters) : m), 0
    ) + 40;

    const ladder = new Float32Array(MAX_STACK);
    const rungs = aglLadder.slice(0, MAX_STACK);
    for (let k = 0; k < MAX_STACK; k++) ladder[k] = rungs[Math.min(k, rungs.length - 1)];
    const hx = 0.5 / (this.meta.tile.width * cols) * cols;
    const hy = 0.5 / (this.meta.tile.height * rows) * rows;
    const t = this.meta.terrain;
    return {
      len: L, tileOff, uvScale, wScale, wFactor, height, isAgl, aglTop,
      ladder, ladderLen: rungs.length, ladderBase: rungs[0] ?? 0,
      tileScale: [1 / cols, 1 / rows],
      clampMin: [hx, hy],
      clampMax: [1 - hx, 1 - hy],
      terrOff: t ? [(t.index % cols) / cols, Math.floor(t.index / cols) / rows] : [0, 0],
      terrRange: t ? [t.hMin, t.hMax] : [0, 0],
    };
  }

  setStackUniforms(gl, U, s) {
    gl.uniform1i(U.u_stackLen, s.len);
    gl.uniform2fv(U.u_tileOff, s.tileOff);
    gl.uniform4fv(U.u_uvScale, s.uvScale);
    gl.uniform2fv(U.u_wScale, s.wScale);
    gl.uniform1fv(U.u_wFactor, s.wFactor);
    gl.uniform1fv(U.u_height, s.height);
    gl.uniform1fv(U.u_isAgl, s.isAgl);
    gl.uniform1f(U.u_aglTop, s.aglTop);
    gl.uniform1fv(U.u_aglLadder, s.ladder);
    gl.uniform1i(U.u_ladderLen, s.ladderLen);
    gl.uniform1f(U.u_useLadder, 1.0);
    gl.uniform2fv(U.u_tileScale, s.tileScale);
    gl.uniform2fv(U.u_clampMin, s.clampMin);
    gl.uniform2fv(U.u_clampMax, s.clampMax);
    gl.uniform2fv(U.u_terrOff, s.terrOff);
    gl.uniform2fv(U.u_terrRange, s.terrRange);
  }

  // Both passes must resolve terrain identically or particles would be drawn
  // at a different height than they were advected at.
  setTerrainUniforms(gl, U, unit) {
    const hi = this.meta.terrainHi;
    const tex = this.frames?.terrainTex;
    gl.uniform1f(U.u_hasTerrHi, tex ? 1.0 : 0.0);
    gl.uniform1i(U.u_terrHi, unit);
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex || this.blankTex);
    gl.activeTexture(gl.TEXTURE0);
    if (tex) gl.uniform2f(U.u_terrHiRange, hi.hMin, hi.hMax);
  }

  setTuningUniforms(gl, U) {
    const tp = this.tuning;
    const hiRes = !!this.frames?.terrainTex;
    const tSrc = hiRes ? this.meta.terrainHi : this.meta.tile;
    gl.uniform1f(U.u_tp, this.terrainPhysics ? tp.gain : 0.0);
    gl.uniform2f(U.u_terrTexel, 1 / tSrc.width, 1 / tSrc.height);
    gl.uniform1f(U.u_oroDecayH, tp.oroDecayH);
    gl.uniform1f(U.u_gammaS, tp.gammaS);
    gl.uniform1f(U.u_gammaC, tp.gammaC);
    gl.uniform1f(U.u_slopeScale, hiRes ? tp.slopeScaleHi : tp.slopeScale);
    gl.uniform1f(U.u_curvScale, tp.curvScale);
    gl.uniform1f(U.u_curvLength, tp.curvLength);
    gl.uniform1f(U.u_ryanGain, tp.ryanGain);
    gl.uniform1f(U.u_leeGain, tp.leeGain);
    gl.uniform1f(U.u_leeDistM, tp.leeDistM);
  }

  // Respawn / mesh window follows the viewport (see the parent windLayer.js
  // for the full derivation and its many corrected failure modes).
  spawnBounds() {
    const b = this.meta.bounds;
    const lonSpan = b.east - b.west;
    const latSpan = b.north - b.south;
    try {
      const mb = this.map.getBounds();
      let x0 = (mb.getWest() - b.west) / lonSpan;
      let x1 = (mb.getEast() - b.west) / lonSpan;
      let y0 = (b.north - mb.getNorth()) / latSpan;
      let y1 = (b.north - mb.getSouth()) / latSpan;

      const cam = this.map.unproject([
        this.map.getCanvas().clientWidth / 2,
        this.map.getCanvas().clientHeight * 0.5,
      ]);
      const cx = (cam.lng - b.west) / lonSpan;
      const cy = (b.north - cam.lat) / latSpan;
      const W = this.map.getCanvas().clientWidth;
      const H = this.map.getCanvas().clientHeight;
      const bl = this.map.unproject([0, H]);
      const br = this.map.unproject([W, H]);
      const nearPt = this.map.unproject([W / 2, H]);
      const acrossDeg = Math.abs(br.lng - bl.lng) / lonSpan;
      const upDeg = Math.abs(cam.lat - nearPt.lat) / latSpan;
      const kmPerLonUnit = lonSpan * 111.32 * Math.cos((cam.lat * Math.PI) / 180);
      const kmPerLatUnit = latSpan * 110.54;
      const reachKm = Math.max(acrossDeg * kmPerLonUnit, upDeg * kmPerLatUnit, 0.2) * 1.6;
      const rx = reachKm / kmPerLonUnit;
      const ry = reachKm / kmPerLatUnit;
      x0 = Math.max(x0, cx - rx); x1 = Math.min(x1, cx + rx);
      y0 = Math.max(y0, cy - ry); y1 = Math.min(y1, cy + ry);
      if (!(x1 > x0 && y1 > y0)) {
        x0 = Math.max(0, (mb.getWest() - b.west) / lonSpan);
        x1 = Math.min(1, (mb.getEast() - b.west) / lonSpan);
        y0 = Math.max(0, (b.north - mb.getNorth()) / latSpan);
        y1 = Math.min(1, (b.north - mb.getSouth()) / latSpan);
      }
      const padX = (x1 - x0) * 0.15, padY = (y1 - y0) * 0.15;
      x0 = Math.max(0, x0 - padX); x1 = Math.min(1, x1 + padX);
      y0 = Math.max(0, y0 - padY); y1 = Math.min(1, y1 + padY);
      if (x1 > x0 && y1 > y0) return { min: [x0, y0], max: [x1, y1] };
    } catch { /* fall through */ }
    return { min: [0, 0], max: [1, 1] };
  }

  // Frame pair for a sim step: both textures must actually be resident —
  // integrating against the fallback texture would write the wrong hour's
  // snowfall into the accumulation state.
  pairExact(t) {
    const leads = this.frames.leads;
    let i = leads.findIndex((l, k) => k + 1 >= leads.length || leads[k + 1] > t);
    i = Math.max(0, Math.min(leads.length - 2, i));
    const a = leads[i], b = leads[i + 1];
    this.frames.ensure(a);
    this.frames.ensure(b);
    const texA = this.frames.textures.get(a);
    const texB = this.frames.textures.get(b);
    if (!texA || !texB) return null;
    this.frames.touch(a);
    this.frames.touch(b);
    const span = Math.max(b - a, 1e-6);
    return { texA, texB, mix: Math.min(1, Math.max(0, (t - a) / span)) };
  }

  // --- per-frame render ------------------------------------------------------

  render(gl, matrix) {
    if (!this.frames) return;
    const b = this.meta.bounds;
    const lonSpan = b.east - b.west;
    const latSpan = b.north - b.south;
    const spawn = this.spawnBounds();

    const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const prevVp = gl.getParameter(gl.VIEWPORT);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);

    // ---- advance the cover simulation toward the target time ----
    const simEnabled = this.floatOK && this.cover && this.snowU;
    if (simEnabled) {
      if (this.time < this.simTime - 0.02) this.resetSim();
      const gap = this.time - this.simTime;
      this.catchingUp = gap > 0.5;
      const maxSteps = this.catchingUp ? 24 : 4;
      let steps = 0;
      while (this.simTime < this.time - 1e-4 && steps < maxSteps) {
        const pair = this.pairExact(this.simTime);
        if (!pair) { this.frames.prefetch(this.simTime, 3); break; }
        const dtH = Math.min(0.1, this.time - this.simTime);
        this.coverStep(gl, pair, dtH, b, lonSpan, latSpan);
        this.simTime += dtH;
        steps++;
      }
      if (this.simTime < this.time - 1e-4) this.map.triggerRepaint();
    } else {
      this.simTime = this.time;
    }

    // Everything visual runs at the sim's clock, so a catch-up replays the
    // storm as a time-lapse instead of teleporting.
    const visT = simEnabled ? Math.min(this.simTime, this.time) : this.time;
    const pairVis = this.frames.getPair(visT);
    this.frames.prefetch(visT);
    if (!pairVis) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
      gl.viewport(prevVp[0], prevVp[1], prevVp[2], prevVp[3]);
      this.map.triggerRepaint();
      return;
    }

    // ---- particle updates ----
    if (this.floatOK && this.snowU) {
      if (this.flakesOn) this.updateParticles(gl, this.flakes, this.flakeProg, this.flakeU, this.flakeStack, pairVis, spawn, b, lonSpan, latSpan, false);
      if (this.streaksOn && this.cover) this.updateParticles(gl, this.streaks, this.streakProg, this.streakU, this.streakStack, pairVis, spawn, b, lonSpan, latSpan, true);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
    gl.viewport(prevVp[0], prevVp[1], prevVp[2], prevVp[3]);

    // ---- draws ----
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    if (this.coverOn) this.drawCover(gl, matrix, pairVis, spawn, b, lonSpan, latSpan);
    if (this.floatOK && this.snowU) {
      if (this.streaksOn && this.cover) {
        this.drawParticles(gl, matrix, this.streaks, this.streakStack, pairVis, b, lonSpan, latSpan, {
          groundLock: 1, streak: 7.0, maxAge: STREAK_MAX_AGE,
          color: [0.97, 0.98, 1.0], speedAlpha: 1, opacity: this.flakeOpacity,
        });
      }
      if (this.flakesOn) {
        this.drawParticles(gl, matrix, this.flakes, this.flakeStack, pairVis, b, lonSpan, latSpan, {
          groundLock: 0, streak: 1.6, maxAge: FLAKE_MAX_AGE,
          color: [0.93, 0.96, 1.0], speedAlpha: 0, opacity: this.flakeOpacity,
        });
      }
    }

    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.map.triggerRepaint();
  }

  coverStep(gl, pair, dtH, b, lonSpan, latSpan) {
    gl.useProgram(this.coverProg);
    gl.bindVertexArray(this.vao);
    const U = this.coverU;
    this.setStackUniforms(gl, U, this.flakeStack);
    this.setSnowUniforms(gl, U);
    this.setTuningUniforms(gl, U);
    gl.uniform1i(U.u_cover, 0);
    gl.uniform1i(U.u_frameA, 2);
    gl.uniform1i(U.u_frameB, 3);
    gl.uniform1f(U.u_frameMix, pair.mix);
    gl.uniform1f(U.u_north, b.north);
    gl.uniform1f(U.u_lonSpan, lonSpan);
    gl.uniform1f(U.u_latSpan, latSpan);
    gl.uniform1f(U.u_dtH, dtH);
    gl.uniform1f(U.u_qGain, this.qGain);
    gl.uniform1f(U.u_meltGain, this.meltGain);
    gl.uniform1f(U.u_windGain, this.windGain);
    gl.uniform2f(U.u_coverTexel, 1 / this.coverW, 1 / this.coverH);
    this.setTerrainUniforms(gl, U, 7);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cover[this.coverCur]);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.coverFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D,
      this.cover[1 - this.coverCur], 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.coverW, this.coverH);
    gl.useProgram(this.coverProg);   // MapLibre may interleave its own GL work
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.coverCur = 1 - this.coverCur;
  }

  updateParticles(gl, sys, prog, U, stack, pair, spawn, b, lonSpan, latSpan, isStreak) {
    gl.useProgram(prog);
    gl.bindVertexArray(this.vao);
    this.setStackUniforms(gl, U, stack);
    this.setSnowUniforms(gl, U);
    this.setTuningUniforms(gl, U);
    gl.uniform1i(U.u_statePos, 0);
    gl.uniform1i(U.u_stateAux, 1);
    gl.uniform1i(U.u_frameA, 2);
    gl.uniform1i(U.u_frameB, 3);
    gl.uniform1f(U.u_frameMix, pair.mix);
    gl.uniform1f(U.u_north, b.north);
    gl.uniform1f(U.u_lonSpan, lonSpan);
    gl.uniform1f(U.u_latSpan, latSpan);
    const zoomFactor = Math.min(1, Math.pow(1.6, 4 - this.map.getZoom()));
    const dtSeconds = (isStreak ? 60.0 : 90.0) * this.speedFactor * Math.max(zoomFactor, 0.03);
    gl.uniform1f(U.u_dt, dtSeconds);
    gl.uniform1f(U.u_windGain, this.windGain);
    gl.uniform1f(U.u_maxAge, isStreak ? STREAK_MAX_AGE : FLAKE_MAX_AGE);
    gl.uniform1f(U.u_time, (performance.now() % 100000) / 1000);
    gl.uniform2fv(U.u_spawnMin, spawn.min);
    gl.uniform2fv(U.u_spawnMax, spawn.max);
    if (isStreak) {
      gl.uniform1i(U.u_cover, 8);
      gl.uniform1f(U.u_saltThresh, this.saltThresh);
      gl.activeTexture(gl.TEXTURE8);
      gl.bindTexture(gl.TEXTURE_2D, this.cover[this.coverCur]);
      gl.activeTexture(gl.TEXTURE0);
    } else {
      gl.uniform1f(U.u_rateRef, 2.5);
      gl.uniform1f(U.u_restFrames, 20.0);
      const hiRes = !!this.frames?.terrainTex;
      const cellKm = hiRes ? 2.1 : 12.7;
      const stepKm = (30.0 * dtSeconds) / 1000.0;
      gl.uniform1i(U.u_substeps, this.terrainPhysics
        ? Math.max(1, Math.min(4, Math.ceil(stepKm / cellKm)))
        : 1);
    }
    this.setTerrainUniforms(gl, U, 7);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.pos);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.aux);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    gl.bindFramebuffer(gl.FRAMEBUFFER, sys.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sys.prevState.pos, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, sys.prevState.aux, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, sys.size, sys.size);
    gl.useProgram(prog);   // rebind: MapLibre can interleave its own GL work
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    sys.swap();
  }

  cameraInfo(b, lonSpan, latSpan) {
    let camX = 0.5, camY = 0.5, camAlt = 0;
    try {
      const cam = this.map.transform.getCameraPosition();
      camAlt = cam.altitude;
      camX = (cam.lngLat.lng - b.west) / lonSpan;
      camY = (b.north - cam.lngLat.lat) / latSpan;
    } catch { camAlt = 0; }
    return { camX, camY, camAlt };
  }

  fadeWindow(spawn, lonSpan, latSpan) {
    const center = this.map.getCenter();
    let viewKm = 40;
    try {
      const cw = this.map.getCanvas().clientWidth;
      const ch = this.map.getCanvas().clientHeight;
      const nearPt = this.map.unproject([cw / 2, ch]);
      const midPt = this.map.unproject([cw / 2, ch * 0.35]);
      const dLatKm = Math.abs(midPt.lat - nearPt.lat) * 110.54;
      const dLonKm = Math.abs(midPt.lng - nearPt.lng) * 111.32
        * Math.cos((center.lat * Math.PI) / 180);
      viewKm = Math.max(Math.hypot(dLatKm, dLonKm), 8);
    } catch { /* keep the default */ }
    const halfBoxKm = 0.5 * Math.max(
      (spawn.max[0] - spawn.min[0]) * lonSpan * 111.32
        * Math.cos((center.lat * Math.PI) / 180),
      (spawn.max[1] - spawn.min[1]) * latSpan * 110.54,
    );
    const cutoff = Math.max(viewKm + halfBoxKm, 2);
    return { near: cutoff * 0.6, far: cutoff * 1.05 };
  }

  mercPerMeter() {
    const center = this.map.getCenter();
    return maplibregl.MercatorCoordinate.fromLngLat(center, 1).z;
  }

  surfaceLift() {
    const center = this.map.getCenter();
    const mpp = 156543.03392 * Math.cos((center.lat * Math.PI) / 180)
      / Math.pow(2, this.map.getZoom());
    return Math.min(4, Math.max(0.5, mpp * 0.5));
  }

  drawParticles(gl, matrix, sys, stack, pair, b, lonSpan, latSpan, opts) {
    gl.useProgram(this.drawProg);
    const D = this.drawU;
    this.setStackUniforms(gl, D, stack);
    gl.uniformMatrix4fv(D.u_matrix, false, matrix);
    gl.uniform1i(D.u_statePosCurr, 0);
    gl.uniform1i(D.u_statePosPrev, 1);
    gl.uniform1i(D.u_stateAuxCurr, 2);
    gl.uniform1i(D.u_frameA, 5);
    gl.uniform1i(D.u_frameB, 6);
    gl.uniform1f(D.u_frameMix, pair.mix);
    gl.uniform1f(D.u_west, b.west);
    gl.uniform1f(D.u_north, b.north);
    gl.uniform1f(D.u_lonSpan, lonSpan);
    gl.uniform1f(D.u_latSpan, latSpan);
    gl.uniform1f(D.u_altMerc, this.mercPerMeter() * this.altScale);
    gl.uniform1f(D.u_streak, opts.streak);
    gl.uniform1f(D.u_maxAge, opts.maxAge);
    gl.uniform1f(D.u_windGain, this.windGain);
    gl.uniform1f(D.u_groundLock, opts.groundLock);
    gl.uniform1f(D.u_ladderBase, stack.ladderBase);
    gl.uniform1f(D.u_surfaceLift, this.surfaceLift());
    gl.uniform1f(D.u_restFrames, 20.0);
    gl.uniform1f(D.u_speedAlpha, opts.speedAlpha);
    gl.uniform1f(D.u_saltThresh, this.saltThresh);
    gl.uniform3fv(D.u_color, opts.color);
    gl.uniform1f(D.u_opacity, opts.opacity);
    gl.uniform1i(D.u_stateSize, sys.size);
    this.setTerrainUniforms(gl, D, 7);

    const spawn = this.spawnBounds();
    const { camX, camY, camAlt } = this.cameraInfo(b, lonSpan, latSpan);
    gl.uniform2f(D.u_camPos, camX, camY);
    gl.uniform1f(D.u_camAlt, camAlt);
    gl.uniform1f(D.u_occlude, this.depthOcclusion && camAlt > 0 ? 1.0 : 0.0);
    const fw = this.fadeWindow(spawn, lonSpan, latSpan);
    gl.uniform1f(D.u_fadeNear, fw.near);
    gl.uniform1f(D.u_fadeFar, fw.far);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.pos);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, sys.prevState.pos);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.aux);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    gl.useProgram(this.drawProg);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.LINES, 0, sys.size * sys.size * 2);
  }

  drawCover(gl, matrix, pair, spawn, b, lonSpan, latSpan) {
    gl.useProgram(this.meshProg);
    const M = this.meshU;
    this.setStackUniforms(gl, M, this.flakeStack);
    this.setSnowUniforms(gl, M);
    this.setTuningUniforms(gl, M);
    gl.uniformMatrix4fv(M.u_matrix, false, matrix);
    gl.uniform1i(M.u_cover, 0);
    gl.uniform1i(M.u_depthRamp, 4);
    gl.uniform1i(M.u_frameA, 5);
    gl.uniform1i(M.u_frameB, 6);
    gl.uniform1f(M.u_frameMix, pair.mix);
    gl.uniform1f(M.u_west, b.west);
    gl.uniform1f(M.u_north, b.north);
    gl.uniform1f(M.u_lonSpan, lonSpan);
    gl.uniform1f(M.u_latSpan, latSpan);
    gl.uniform2fv(M.u_boxMin, spawn.min);
    gl.uniform2fv(M.u_boxMax, spawn.max);
    gl.uniform1i(M.u_grid, this.meshGrid);
    gl.uniform1f(M.u_altMerc, this.mercPerMeter() * this.altScale);
    gl.uniform1f(M.u_surfaceLift, this.surfaceLift() * 0.6);
    gl.uniform1f(M.u_mode, this.modelDepthMode || !this.floatOK ? 1.0 : 0.0);
    gl.uniform1f(M.u_opacity, this.coverOpacity);
    gl.uniform1f(M.u_depthMax, this.depthMax);
    gl.uniform1f(M.u_dispGain, this.dispGain);
    gl.uniform1f(M.u_windGain, this.windGain);

    const { camX, camY, camAlt } = this.cameraInfo(b, lonSpan, latSpan);
    gl.uniform2f(M.u_camPos, camX, camY);
    gl.uniform1f(M.u_camAlt, camAlt);
    gl.uniform1f(M.u_occlude, this.depthOcclusion && camAlt > 0 ? 1.0 : 0.0);
    const fw = this.fadeWindow(spawn, lonSpan, latSpan);
    gl.uniform1f(M.u_fadeNear, fw.near);
    gl.uniform1f(M.u_fadeFar, fw.far * 1.15);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cover ? this.cover[this.coverCur] : this.blankTex);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    gl.useProgram(this.meshProg);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, this.meshGrid * this.meshGrid * 6);
  }
}
