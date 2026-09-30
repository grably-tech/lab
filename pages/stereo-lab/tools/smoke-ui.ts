// Headless UI smoke test: dataset run + scene screenshots at two frames, then live mode on Chrome's fake cameras.
// Usage: bun --install=force run tools/smoke-ui.ts <outDir>   (server from tools/serve.ts must be running)
import { join } from "node:path";
import { chromium } from "playwright-core";

const out = process.argv[2];
const vendor = join(import.meta.dir, "..", "data", "vendor");
const browser = await chromium.launch({
  executablePath: `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
  headless: true,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream=device-count=2"],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
// Same CDN workaround as detect-all.ts: headless Chrome here drops CDN connections, so serve MediaPipe from data/vendor.
await page.route(/^https:\/\/(cdn\.jsdelivr\.net\/npm\/@mediapipe\/tasks-vision@1\.0\.1|storage\.googleapis\.com\/mediapipe-models)\//, (route) => {
  const path = new URL(route.request().url()).pathname;
  const file = path.endsWith(".task") ? path.slice(path.lastIndexOf("/") + 1) : path.replace(/^.*tasks-vision@1\.0\.1\//, "");
  const type = file.endsWith(".wasm") ? "application/wasm" : file.endsWith(".task") ? "application/octet-stream" : "text/javascript";
  return route.fulfill({ path: join(vendor, file), headers: { "access-control-allow-origin": "*", "content-type": type } });
});

await page.goto("http://localhost:5180/?delegate=CPU&mode=dataset");
await page.waitForFunction(() => "stereoLab" in window, null, { timeout: 60000 });
await page.selectOption("#dsB", "00_29");
await page.click("#dsRun");
await page.waitForFunction(() => /^Готово|^Ошибка/.test(document.getElementById("status")!.textContent!), null, { timeout: 180000 });
console.log("dataset:", await page.textContent("#status"));
for (const frame of ["20", "200"]) {
  await page.fill("#scrub", frame);
  await page.dispatchEvent("#scrub", "input");
  await page.waitForTimeout(1200);
  await page.locator(".scene:has(#scene)").screenshot({ path: join(out, `scene_${frame}.png`) });
}
await page.screenshot({ path: join(out, "ui_dataset.png") });

await page.goto("http://localhost:5180/?delegate=CPU");
await page.waitForTimeout(1500);
await page.click("#liveOpen");
await page.waitForTimeout(6000);
console.log("live:", await page.textContent("#status"), "| reset enabled:", await page.isEnabled("#liveReset"));
await page.screenshot({ path: join(out, "ui_live.png") });

// Calibration persistence: a synthetic record for the open fake cameras must be picked up on reopen,
// exported as-is, and imports for other cameras or of junk must be rejected with a visible error.
const record = await page.evaluate(() => {
  const dev = (sel: string, vid: string) => {
    const v = document.getElementById(vid) as HTMLVideoElement;
    return { label: (document.getElementById(sel) as HTMLSelectElement).selectedOptions[0].textContent, width: v.videoWidth, height: v.videoHeight };
  };
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const r = {
    format: "stereo-lab-calibration", version: 2, createdAt: new Date().toISOString(),
    cameras: { a: dev("liveA", "videoA"), b: dev("liveB", "videoB") },
    camA: { f: 900, cx: 320, cy: 240, R: I, t: [0, 0, 0] }, camB: { f: 950, cx: 320, cy: 240, R: I, t: [1, 0, 0] },
    view: { center: [0, 0, 3], radius: 1 }, medianReprojPx: 1.5, faceWidthUnits: 0.3, log: ["synthetic"],
  };
  localStorage.setItem("stereo-lab.calibration", JSON.stringify(r));
  return r;
});
await page.click("#liveOpen");
await page.waitForTimeout(5000);
console.log("reopen:", await page.textContent("#status"));
console.log("saved info:", await page.textContent("#savedInfo"));
const [download] = await Promise.all([page.waitForEvent("download"), page.click("#liveExport")]);
const exported = JSON.parse(await Bun.file(await download.path()).text());
console.log("export matches stored:", JSON.stringify(exported) === JSON.stringify(record), download.suggestedFilename());
const importText = async (text: string) => {
  await page.setInputFiles("#liveImportFile", { name: "c.json", mimeType: "application/json", buffer: Buffer.from(text) });
  await page.waitForTimeout(800);
  return page.textContent("#status");
};
console.log("import other camera:", await importText(JSON.stringify({ ...record, cameras: { ...record.cameras, b: { ...record.cameras.b, label: "Other Camera" } } })));
console.log("import junk:", await importText("{ not json"));
console.log("import wrong format:", await importText(JSON.stringify({ hello: 1 })));
console.log("import valid:", await importText(JSON.stringify(record)));
await browser.close();
