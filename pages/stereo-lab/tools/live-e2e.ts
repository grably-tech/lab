// Live-mode end-to-end on real hand footage: two InterHand2.6M clips are served as the two "cameras" (getUserMedia is
// replaced by canvas streams of looping videos — Chrome's file-fed fake capture exposes only one device). Calibrates,
// then blanks camera B and checks the hand stays in 3D as a single-camera placement.
// Usage: bun --install=force run tools/live-e2e.ts <outDir> [camA] [camB]   (server from tools/serve.ts must be running)
import { join } from "node:path";
import { chromium } from "playwright-core";

const [out, camA = "400285", camB = "400289"] = process.argv.slice(2);
const vendor = join(import.meta.dir, "..", "data", "vendor");

const browser = await chromium.launch({
  executablePath: `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"],
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
await page.addInitScript(({ clips }) => {
  // __blank[id]: the camera sees nothing; __rollDeg[id]: the camera is rolled by that many degrees (a bumped camera).
  const w = window as unknown as { __blank: Record<string, boolean>; __rollDeg: Record<string, number> };
  w.__blank = {};
  w.__rollDeg = {};
  const devices = Object.keys(clips).map((id) => ({ deviceId: id, groupId: id, kind: "videoinput", label: `clip ${id}`, toJSON() {} }));
  const videos = new Map<string, HTMLVideoElement>();
  const clipVideo = (id: string) => {
    if (!videos.has(id)) {
      const v = document.createElement("video");
      Object.assign(v, { muted: true, loop: true, playsInline: true, preload: "auto", src: clips[id] });
      v.load();
      videos.set(id, v);
    }
    return videos.get(id)!;
  };
  const stream = async (id: string) => {
    const video = clipVideo(id);
    if (video.readyState < 2) await new Promise((r) => video.addEventListener("loadeddata", r, { once: true }));
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d")!;
    // Redraw only on new clip frames, so the stream carries the clip's real frame rate like a camera would.
    const draw = () => {
      if (w.__blank[id]) ctx.fillRect(0, 0, canvas.width, canvas.height);
      else {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(((w.__rollDeg[id] ?? 0) * Math.PI) / 180);
        ctx.drawImage(video, -canvas.width / 2, -canvas.height / 2);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
      }
      video.requestVideoFrameCallback(draw);
    };
    draw();
    return canvas.captureStream(30);
  };
  navigator.mediaDevices.enumerateDevices = async () => devices as unknown as MediaDeviceInfo[];
  navigator.mediaDevices.getUserMedia = async (c?: MediaStreamConstraints) => {
    const id = ((c?.video as MediaTrackConstraints)?.deviceId as { exact?: string })?.exact ?? devices[0].deviceId;
    return stream(id);
  };
  // Both clips start together once both cameras are open, so the two views stay in sync.
  (window as unknown as { __startClips: () => void }).__startClips = () => videos.forEach((v) => ((v.currentTime = 0), v.play()));
}, { clips: { [camA]: `/data/interhand/clip/cam_${camA}.mp4`, [camB]: `/data/interhand/clip/cam_${camB}.mp4` } });

page.on("console", (m) => m.type() === "error" && console.log("[console]", m.text()));
await page.goto("http://localhost:5180/?delegate=CPU");
await page.waitForFunction(() => "stereoLab" in window, null, { timeout: 60000 });
await page.evaluate(() => localStorage.clear());
await page.reload();
await page.waitForFunction(() => "stereoLab" in window, null, { timeout: 60000 });
await page.waitForFunction(() => document.querySelectorAll("#liveA option").length === 2);
await page.selectOption("#liveA", camA);
await page.selectOption("#liveB", camB);
await page.click("#liveOpen");
await page.waitForFunction(() => /Калибруюсь|Подхватил|Камеры:/.test(document.getElementById("status")!.textContent!), null, { timeout: 60000 });
await page.evaluate(() => (window as unknown as { __startClips: () => void }).__startClips());
// No button: auto-calibration starts once enough varied frames are in; wait until it has been checked on new frames.
await page.waitForFunction(() => /^Калибровка сходится|не удалась/.test(document.getElementById("status")!.textContent!), null, { timeout: 240000 });
console.log("calibration:", await page.textContent("#status"));
console.log((await page.textContent("#metrics"))!.replace(/\s+/g, " "));

const legend = () => page.textContent("#legend");
await page.waitForTimeout(3000);
console.log("both cameras:", await legend());
console.log("single camera panel:", await page.textContent("#legendMono"));
await page.screenshot({ path: join(out, "live_stereo.png") });
await page.evaluate((id) => ((window as unknown as { __blank: Record<string, boolean> }).__blank[id] = true), camB);
const singles: string[] = [];
for (let i = 0; i < 6; i++) {
  await page.waitForTimeout(500);
  singles.push((await legend())!);
}
console.log("camera B blank:", singles.map((l) => (/только камера A/.test(l) ? "single A" : l)).join(" | "));
await page.screenshot({ path: join(out, "live_single.png") });
await page.evaluate((id) => ((window as unknown as { __blank: Record<string, boolean> }).__blank[id] = false), camB);
await page.waitForTimeout(1500);
console.log("camera B back:", await legend());
// Bump camera B: the calibration must notice and replace itself.
await page.evaluate((id) => ((window as unknown as { __rollDeg: Record<string, number> }).__rollDeg[id] = 4), camB);
await page.waitForFunction(() => /сдвинули/.test(document.getElementById("status")!.textContent!), null, { timeout: 60000 });
console.log("camera B rolled 4°:", await page.textContent("#status"));
await page.waitForFunction(() => /^Калибровка сходится|не удалась/.test(document.getElementById("status")!.textContent!), null, { timeout: 240000 });
console.log("after recalibration:", await page.textContent("#status"));
await browser.close();
