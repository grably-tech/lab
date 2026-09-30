// Single-camera placement on saved landmarks: per pair, calibrate, then hide one camera's detections and place the
// hand from the other camera with depth anchored g frames earlier by stereo. Errors use the similarity fitted to the
// stereo reconstruction (not refitted to the mono points), so depth errors are not aligned away.
// Usage: bun run tools/eval-mono.ts <interhand|panoptic> [step] [oracle]   — oracle: GT calibration instead of ours.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { DATASETS, GT_SKELETONS, gtPairCameras, normalizeGt, poseErrors } from "../eval.js";
import { calibrationFrames, emptyPoints, LAYOUTS } from "../landmarks.js";
import { fuseFrame, groupsOf, stereoDepths } from "../mono.js";
import { calibratePair, median, reconstruct } from "../stereo.js";

type Frame = { image: number[][]; world: (number[] | null)[] } | null;
const [datasetKey = "interhand", stepArg = "1", mode] = process.argv.slice(2);
const GAPS = [0, 1, 5, 10, 25];
const dataset = DATASETS[datasetKey as keyof typeof DATASETS];
const layout = LAYOUTS[dataset.layout as keyof typeof LAYOUTS];
const groups = groupsOf(layout);
const { jointMap } = GT_SKELETONS[dataset.skeleton as keyof typeof GT_SKELETONS];
const step = Number(stepArg);
const dir = join(import.meta.dir, "..", dataset.dir);
const gt = normalizeGt(JSON.parse(readFileSync(join(dir, "clip", "gt.json"), "utf8")));
const landmarks: Record<string, Frame[]> = JSON.parse(readFileSync(join(dir, "landmarks", `step${step}.json`), "utf8"));
const cams = Object.keys(gt.cameras);
const hidden = emptyPoints(layout.size);
const errors: Record<string, number[]> = {};
const add = (key: string, values: number[]) => (errors[key] ??= []).push(...values);

for (const b of cams.slice(1)) {
  const a = cams[0];
  const la = landmarks[`${a}@${step}`];
  const lb = landmarks[`${b}@${step}`];
  const idx = la.map((_, i) => i).filter((i) => la[i] && lb[i]);
  const frames = idx.map((i) => ({ a: la[i]!.image, b: lb[i]!.image, shapeA: la[i]!.world, shapeB: lb[i]!.world }));
  const size = (n: string) => ({ width: gt.cameras[n].width, height: gt.cameras[n].height });
  const gtJoints = idx.map((i) => gt.joints[i]);
  let calib;
  try {
    calib = mode === "oracle" ? gtPairCameras(gt.cameras[a], gt.cameras[b]) : calibratePair(calibrationFrames(frames, layout), size(a), size(b), { bones: layout.bones, groups: layout.groups });
  } catch (e) {
    console.log(`${a}+${b} | calibration failed: ${(e as Error).message}`);
    continue;
  }
  const recon = reconstruct(frames, calib);
  const { seqFit } = poseErrors(recon, gtJoints, jointMap);
  const frameErrors = (points: (number[] | null)[], k: number) =>
    jointMap.filter(([m, g]) => points[m] && gtJoints[k][g][3] > 0).map(([m, g]) => Math.hypot(...seqFit.apply(points[m]!).map((v, i) => v - gtJoints[k][g][i])));
  const anchors = frames.map((f, k) => stereoDepths(f, recon[k], groups, calib));
  frames.forEach((f, k) => add("stereo", frameErrors(recon[k], k)));
  for (const g of GAPS) {
    for (let k = g; k < frames.length; k++) {
      for (const onlyA of [true, false]) {
        const frame = { ...frames[k], a: onlyA ? frames[k].a : hidden, b: onlyA ? hidden : frames[k].b };
        add(`one camera, anchor ${g} fr ago`, frameErrors(fuseFrame(frame, recon[k], calib, anchors[k - g], groups).points, k));
      }
    }
  }
}
const fmt = (v: number[]) => {
  const s = [...v].sort((x, y) => x - y);
  return `median ${median(s).toFixed(1)} cm, p90 ${s[Math.floor(s.length * 0.9)].toFixed(1)} cm (${s.length} joints)`;
};
console.log(`${datasetKey} step ${step}, ${mode === "oracle" ? "GT" : "estimated"} calibration, ${1 / (gt.fps / step)} s per frame`);
for (const [k, v] of Object.entries(errors)) console.log(`  ${k.padEnd(28)} ${fmt(v)}`);
