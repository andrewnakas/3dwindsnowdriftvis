// Loads atlas PNGs into GL textures with prefetch + LRU eviction.

// Each cached frame is a ~20 MB GPU texture; keep the cap low for mobile GPUs.
const CACHE_MAX = 4;

// Scratch texture unit for uploads — above every unit the wind shaders bind
// (0-7), so an upload mid-frame cannot disturb a sampler in use.
const UPLOAD_UNIT = 12;

export class FrameManager {
  constructor(gl, meta, basePath = "data/") {
    this.gl = gl;
    this.meta = meta;
    this.basePath = basePath;
    this.textures = new Map(); // lead -> WebGLTexture
    this.loading = new Map();  // lead -> Promise
    this.lru = [];
    this.version = encodeURIComponent(meta.init_time);
    this.leads = meta.frames.map((f) => f.lead_hours);
  }

  nearestLead(t) {
    return this.leads.reduce((a, b) => (Math.abs(b - t) < Math.abs(a - t) ? b : a));
  }

  // Continuous time t (hours) -> pair of frame textures + mix, using whatever
  // is loaded; kicks off loads for what's missing. Handles non-uniform lead
  // spacing (e.g. sparse test builds).
  getPair(t) {
    let i = this.leads.findIndex((l, k) => k + 1 >= this.leads.length || this.leads[k + 1] > t);
    i = Math.max(0, Math.min(this.leads.length - 2, i));
    const a = this.leads[i];
    const b = this.leads[i + 1];
    this.ensure(a);
    this.ensure(b);
    const texA = this.textures.get(a);
    const texB = this.textures.get(b);
    if (texA && texB) {
      this.touch(a); this.touch(b);
      const span = Math.max(b - a, 1e-6);
      return { texA, texB, mix: Math.max(0, Math.min(1, (t - a) / span)) };
    }
    const any = texA || texB || this.newestLoaded();
    return any ? { texA: any, texB: any, mix: 0 } : null;
  }

  prefetch(t, count = 2) {
    const i = Math.floor(t);
    for (let k = 1; k <= count; k++) {
      const idx = i + k;
      if (idx < this.leads.length) this.ensure(this.leads[idx]);
    }
  }

  newestLoaded() {
    return this.lru.length ? this.textures.get(this.lru[this.lru.length - 1]) : null;
  }

  ensure(lead) {
    if (this.textures.has(lead) || this.loading.has(lead)) return;
    const entry = this.meta.frames.find((f) => f.lead_hours === lead);
    if (!entry) return;
    const url = `${this.basePath}${entry.file}?v=${this.version}`;
    const p = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
        return r.blob();
      })
      .then((blob) =>
        // premultiplyAlpha option is unsupported on some Safari versions;
        // premultiplication only affects alpha=0 (invalid) texels, so plain
        // decode is an acceptable fallback.
        createImageBitmap(blob, { premultiplyAlpha: "none" }).catch(() => createImageBitmap(blob))
      )
      .then((bmp) => {
        this.textures.set(lead, this.upload(bmp));
        bmp.close();
        this.touch(lead);
        this.evict();
      })
      .catch((e) => console.error("frame load failed", lead, e))
      .finally(() => this.loading.delete(lead));
    this.loading.set(lead, p);
  }

  upload(bitmap) {
    const gl = this.gl;
    const tex = gl.createTexture();
    // Uploads land between frames on whichever unit happened to be active, and
    // unbinding at the end would leave that unit empty — if it was one the
    // wind shaders sample from, the next draw is INVALID_OPERATION. Do the
    // work on a high scratch unit nothing samples, and leave unit 0 active.
    gl.activeTexture(gl.TEXTURE0 + UPLOAD_UNIT);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.activeTexture(gl.TEXTURE0);
    return tex;
  }

  // The hi-res terrain texture is time-invariant, so it loads once and stays
  // outside the LRU. onLoad fires with the texture; failure is non-fatal —
  // the caller falls back to the coarse terrain tile inside the atlas.
  loadTerrain(onLoad) {
    const hi = this.meta.terrainHi;
    if (!hi?.file) return;
    const url = `${this.basePath}${hi.file}?v=${this.version}`;
    // Halving the texture on memory-constrained devices costs terrain detail
    // but keeps the physics working where it would otherwise blow the budget.
    const half = !!sessionStorage.getItem("lowmem");
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
        // arrayBuffer rather than blob: at this texture's size (~15 MB, 42 MB
        // decoded) blob() fails outright in some Chrome builds.
        return r.arrayBuffer();
      })
      .then((buf) => {
        const blob = new Blob([buf], { type: "image/png" });
        const opts = { premultiplyAlpha: "none" };
        if (half) Object.assign(opts, { resizeWidth: hi.width >> 1, resizeHeight: hi.height >> 1 });
        return createImageBitmap(blob, opts).catch(() => createImageBitmap(blob));
      })
      .then((bmp) => {
        // Build the texture in a local first: `upload` rebinds texture units
        // and leaves this one bound, so publishing the field before it returns
        // lets a render in the same tick bind a texture that is not yet
        // complete, which makes the whole draw INVALID_OPERATION.
        const tex = this.upload(bmp);
        bmp.close();
        this.terrainTex = tex;
        onLoad?.(tex);
      })
      .catch((e) => console.warn("hi-res terrain unavailable, using atlas tile:", e.message));
  }

  touch(lead) {
    const i = this.lru.indexOf(lead);
    if (i >= 0) this.lru.splice(i, 1);
    this.lru.push(lead);
  }

  evict() {
    while (this.lru.length > CACHE_MAX) {
      const lead = this.lru.shift();
      const tex = this.textures.get(lead);
      if (tex) this.gl.deleteTexture(tex);
      this.textures.delete(lead);
    }
  }
}
