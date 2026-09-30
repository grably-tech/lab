// Robustness to whole-frame garbage: corrupt a share of camera-B frames in real Panoptic landmarks and
// measure how the calibration-only pipeline degrades (3D error vs GT; corrupted frames are excluded from scoring).
// Usage: bun run tools/robustness.ts [step]
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { GT_SKELETONS, normalizeGt, poseErrors } from "../eval.js";
import { BODY_BONES } from "../landmarks.js";
import { calibratePair, mulberry32, reconstruct } from "../stereo.js";

type Pts = number[][];
const step = Number(process.argv[2] ?? 2);
const data = join(import.meta.dir, "..", "data", "panoptic-pose1");
const gt = normalizeGt(JSON.parse(readFileSync(join(data, "clip", "gt.json"), "utf8")));
const lm = JSON.parse(readFileSync(join(data, "landmarks", `step${step}.json`), "utf8"));

const LR = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
const swapLR = (p: Pts) => {
  const q = p.slice();
  for (const [l, r] of LR) [q[l], q[r]] = [p[r], p[l]];
  return q;
};

// Each corruption returns camera-B landmarks for frame k; `pool` is the clean B sequence.
const corruptions: Record<string, (pool: Pts[], k: number, rand: () => number) => Pts> = {
  // MediaPipe mirrored the person (left/right swapped) in one view.
  swapLR: (pool, k) => swapLR(pool[k]),
  // Detection belongs to a different moment (glitch / tracking jump): a plausible pose, but the wrong one.
  wrongPose: (pool, k, rand) => pool[(k + 20 + Math.floor(rand() * (pool.length - 40))) % pool.length],
  // Every joint thrown off by ~40 px (motion blur, partial occlusion guesses).
  noisy40px: (pool, k, rand) => pool[k].map(([u, v, vis]) => [u + (rand() - 0.5) * 80, v + (rand() - 0.5) * 80, vis]),
};

const pairs = [["00_00", "00_29"], ["00_00", "00_21"], ["00_00", "00_11"]];
const size = { width: 960, height: 540 };

function run(a: string, b: string, kind: string | null, rate: number, lagFrames = 0) {
  const la = lm[`${a}@${step}`];
  const lb = lm[`${b}@${step}`];
  const idx = la.map((_: unknown, i: number) => i).filter((i: number) => la[i] && lb[i] && lb[i + lagFrames * step]);
  const clean = idx.map((i: number) => lb[i + lagFrames * step].image as Pts);
  const rand = mulberry32(42);
  const bad = new Set<number>();
  const framesB = clean.map((p: Pts, k: number) => {
    if (!kind || rand() >= rate) return p;
    bad.add(k);
    return corruptions[kind](clean, k, rand);
  });
  const frames = idx.map((i: number, k: number) => ({ a: la[i].image, b: framesB[k], shapeA: la[i].world }));
  const calib = calibratePair(frames, size, size, { bones: BODY_BONES });
  // Score only on clean, in-sync frames so the number reflects the calibration, not the injected garbage itself.
  const good = idx.map((_: number, k: number) => k).filter((k: number) => !bad.has(k));
  const scoreFrames = good.map((k: number) => ({ a: la[idx[k]].image, b: lb[idx[k]].image }));
  const rejected = new Set<number>(calib.rejectedDetections.map(Number));
  const caught = [...bad].filter((k) => rejected.has(k)).length;
  return {
    error: poseErrors(reconstruct(scoreFrames, calib), good.map((k: number) => gt.joints[idx[k]]), GT_SKELETONS.coco19.jointMap).seq.median,
    caught: `${caught}/${bad.size}`,
    falseRejects: rejected.size - caught,
  };
}

type Result = ReturnType<typeof run>;
const fmt = (rs: Result[]) => rs.map((r) => `${r.error.toFixed(1).padStart(5)} cm [bad caught ${r.caught.padStart(7)}, good dropped ${String(r.falseRejects).padStart(2)}]`).join("  ");
console.log(`pairs: ${pairs.map((p) => p.join("+")).join(", ")}`);
console.log(`clean              ${fmt(pairs.map(([a, b]) => run(a, b, null, 0)))}`);
for (const kind of Object.keys(corruptions)) {
  for (const rate of [0.1, 0.2, 0.3]) {
    console.log(`${kind.padEnd(10)} ${String(rate * 100).padStart(3)}%    ${fmt(pairs.map(([a, b]) => run(a, b, kind, rate)))}`);
  }
}
// Camera B lagging behind A for the whole session (Continuity Camera latency); the calibration is fit on lagged
// pairs and scored on in-sync frames, so the number shows how much the lag distorts the camera geometry.
for (const lag of [1, 2, 4]) {
  console.log(`B lags ${String(lag * step * 33).padStart(3)} ms      ${fmt(pairs.map(([a, b]) => run(a, b, null, 0, lag)))}`);
}
