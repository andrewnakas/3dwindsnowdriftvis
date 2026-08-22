// Manual verification: drive the site headlessly and capture screenshots.
//
//   cd site && python3 -m http.server 8123 &
//   node tools/screenshot.mjs http://localhost:8123 /tmp/shots
//
// Requires playwright (npm i -g playwright) and a Chromium it can find
// (PLAYWRIGHT_BROWSERS_PATH or a system install).

import { mkdirSync } from "node:fs";

const base = process.argv[2] ?? "http://localhost:8123";
const outDir = process.argv[3] ?? "/tmp/snow-shots";
mkdirSync(outDir, { recursive: true });

let chromium, executablePath;
try {
  ({ chromium } = await import("playwright"));
} catch {
  ({ chromium } = await import("playwright-core"));
  executablePath = process.env.CHROMIUM_PATH ?? "/opt/pw-browsers/chromium/chrome-linux/chrome";
}

const browser = await chromium.launch({
  executablePath,
  args: [
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--disable-gpu-sandbox",
    "--no-sandbox",
  ],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(`console: ${m.text()}`);
});

await page.goto(base, { waitUntil: "networkidle" });
await page.waitForFunction(() => window.__snow && window.__snow.frames, null, { timeout: 30000 });

// Let the autoplaying storm run a few seconds, then capture.
await page.waitForTimeout(8000);
await page.screenshot({ path: `${outDir}/t-early.png` });

// Pause playback, jump to a later hour, let the sim catch up, capture again.
await page.evaluate(() => {
  document.getElementById("play").click(); // pause if playing
});
for (const t of [4, 8]) {
  await page.evaluate((tt) => { window.__snow.time = tt; }, t);
  await page.waitForFunction(
    (tt) => Math.abs(window.__snow.simTime - Math.min(tt, window.__snow.time)) < 0.01
      || !window.__snow.catchingUp,
    t, { timeout: 30000 }
  );
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${outDir}/t-${t}.png` });
}

const state = await page.evaluate(() => ({
  simTime: window.__snow.simTime,
  time: window.__snow.time,
  floatOK: window.__snow.floatOK,
  hasSnowMeta: !!window.__snow.snowU,
  terrainHi: !!window.__snow.frames.terrainTex,
}));
console.log("state:", JSON.stringify(state));
console.log(errors.length ? `ERRORS (${errors.length}):\n` + errors.slice(0, 20).join("\n") : "no page errors");
await browser.close();
process.exit(errors.length ? 1 : 0);
