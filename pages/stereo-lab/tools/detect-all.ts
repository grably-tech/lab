// Headless run of the real page: MediaPipe on every clip, landmarks saved for offline iteration,
// then every (00_00, X) pair is calibrated in the browser and scored against GT.
// Usage: bun --install=force run tools/detect-all.ts <panoptic|interhand> [step]   (tools/serve.ts must be running)
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright-core";
import { DATASETS } from "../eval.js";

const [key = "panoptic", stepArg = "2"] = process.argv.slice(2);
const step = Number(stepArg);
const base = process.env.LAB_URL ?? "http://localhost:5180/";
const dir = join(import.meta.dir, "..", DATASETS[key as keyof typeof DATASETS].dir);
const cams = Object.keys(JSON.parse(readFileSync(join(dir, "clip", "gt.json"), "utf8")).cameras);
const outDir = join(dir, "landmarks");

const browser = await chromium.launch({
  executablePath: `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
  headless: true,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
page.on("console", (m) => m.type() === "error" && console.log("[console.error]", m.text().slice(0, 300)));
page.on("requestfailed", (r) => console.log("[requestfailed]", r.url(), r.failure()?.errorText));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
// Headless Chrome on this machine intermittently fails CDN fetches with ERR_SOCKET_NOT_CONNECTED while curl works,
// so the harness serves the same MediaPipe release from data/vendor (fetched with curl). The page itself uses the CDN.
const vendor = join(import.meta.dir, "..", "data", "vendor");
await page.route(/^https:\/\/(cdn\.jsdelivr\.net\/npm\/@mediapipe\/tasks-vision@1\.0\.1|storage\.googleapis\.com\/mediapipe-models)\//, (route) => {
  const path = new URL(route.request().url()).pathname;
  const file = path.endsWith(".task") ? path.slice(path.lastIndexOf("/") + 1) : path.replace(/^.*tasks-vision@1\.0\.1\//, "");
  const type = file.endsWith(".wasm") ? "application/wasm" : file.endsWith(".task") ? "application/octet-stream" : "text/javascript";
  return route.fulfill({ path: join(vendor, file), headers: { "access-control-allow-origin": "*", "content-type": type } });
});
await page.goto(`${base}?delegate=CPU&mode=dataset`);
await page.waitForFunction(() => "stereoLab" in window, null, { timeout: 60000 });

for (const cam of cams) {
  const t0 = Date.now();
  await page.evaluate(([k, c, s]) => (window as any).stereoLab.detect(k, c, s), [key, cam, step] as const);
  console.log(`${cam}: detected in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
mkdirSync(outDir, { recursive: true });
const landmarks = await page.evaluate((k) => (window as any).stereoLab.exportLandmarks(k), key);
writeFileSync(join(outDir, `step${step}.json`), JSON.stringify(landmarks));

for (const cam of cams.slice(1)) {
  const r = await page.evaluate(([k, a, b, s]) => (window as any).stereoLab.runPair(k, a, b, s), [key, cams[0], cam, step] as const);
  const c = r.calibration;
  console.log(
    `${cams[0]}+${cam} (${c.angleBetweenCamerasDeg.toFixed(0)}°): FOV ${c.fovA.est.toFixed(1)}/${c.fovA.gt.toFixed(1)} ${c.fovB.est.toFixed(1)}/${c.fovB.gt.toFixed(1)}` +
      ` rot ${c.rotationErrorDeg.toFixed(2)}° t ${c.translationDirErrorDeg.toFixed(2)}°` +
      ` | 3D seq ${r.stereo.seq.median.toFixed(1)} cm, PA ${r.stereo.perFrame.mean.toFixed(1)} cm, cover ${(r.stereo.coverage * 100).toFixed(0)}%` +
      ` | oracle ${r.oracle.seq.median.toFixed(1)}/${r.oracle.perFrame.mean.toFixed(1)} cm | mono PA ${r.mono.mean.toFixed(1)} cm`,
  );
  await page.screenshot({ path: join(outDir, `page_${cams[0]}_${cam}.png`) });
  if (cam === cams[3]) break;
}
await browser.close();
