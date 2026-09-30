// Offline feedback loop: same stereo.js/eval.js as the page, on landmarks saved by tools/detect-all.ts.
// Usage: bun run tools/eval-pairs.ts <panoptic|interhand> [step] [all]   — default pairs the first camera with each other.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { calibrationErrors, DATASETS, GT_SKELETONS, gtRelativePose, normalizeGt, oracleReconstruction, poseErrors } from "../eval.js";
import { calibrationFrames, LAYOUTS } from "../landmarks.js";
import { calibratePair, reconstruct, rotationAngleDeg } from "../stereo.js";

type Frame = { image: number[][]; world: (number[] | null)[] } | null;
const [datasetKey = "panoptic", stepArg = "2", mode] = process.argv.slice(2);
const dataset = DATASETS[datasetKey as keyof typeof DATASETS];
const layout = LAYOUTS[dataset.layout as keyof typeof LAYOUTS];
const { jointMap } = GT_SKELETONS[dataset.skeleton as keyof typeof GT_SKELETONS];
const step = Number(stepArg);
const dir = join(import.meta.dir, "..", dataset.dir);
const gt = normalizeGt(JSON.parse(readFileSync(join(dir, "clip", "gt.json"), "utf8")));
const landmarks: Record<string, Frame[]> = JSON.parse(readFileSync(join(dir, "landmarks", `step${step}.json`), "utf8"));
const cams = Object.keys(gt.cameras);
const pairs = mode === "all" ? cams.flatMap((a, i) => cams.slice(i + 1).map((b) => [a, b])) : cams.slice(1).map((b) => [cams[0], b]);

for (const [a, b] of pairs) {
  const la = landmarks[`${a}@${step}`];
  const lb = landmarks[`${b}@${step}`];
  const idx = la.map((_, i) => i).filter((i) => la[i] && lb[i]);
  const frames = idx.map((i) => ({ a: la[i]!.image, b: lb[i]!.image, shapeA: la[i]!.world }));
  const size = (n: string) => ({ width: gt.cameras[n].width, height: gt.cameras[n].height });
  const gtJoints = idx.map((i) => gt.joints[i]);
  const t0 = performance.now();
  let line: string;
  try {
    const calib = calibratePair(calibrationFrames(frames, layout), size(a), size(b), { bones: layout.bones, groups: layout.groups });
    const c = calibrationErrors(calib, gt.cameras[a], gt.cameras[b]);
    const stereo = poseErrors(reconstruct(frames, calib), gtJoints, jointMap);
    line =
      `FOV ${c.fovA.est.toFixed(1)}/${c.fovA.gt.toFixed(1)} ${c.fovB.est.toFixed(1)}/${c.fovB.gt.toFixed(1)} rot ${c.rotationErrorDeg.toFixed(2)}° t ${c.translationDirErrorDeg.toFixed(2)}°` +
      ` | 3D ${stereo.seq.median.toFixed(2)} cm PA ${stereo.perFrame.mean.toFixed(2)}\n    ${calib.log.join(" | ")}`;
  } catch (e) {
    line = `calibration failed: ${(e as Error).message}`;
  }
  const oracle = poseErrors(oracleReconstruction(frames, gt.cameras[a], gt.cameras[b]), gtJoints, jointMap);
  const mono = poseErrors(idx.map((i) => la[i]!.world), gtJoints, jointMap).perFrame;
  const angle = rotationAngleDeg(gtRelativePose(gt.cameras[a], gt.cameras[b]).R, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  console.log(
    `${a}+${b} ${angle.toFixed(0).padStart(3)}° (${frames.length} fr) | oracle ${oracle.seq.median.toFixed(2)} PA ${oracle.perFrame.mean.toFixed(2)} | mono PA ${mono.mean.toFixed(2)} | ${((performance.now() - t0) / 1000).toFixed(1)} s | ${line}`,
  );
}
