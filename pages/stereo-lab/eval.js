// Ground-truth evaluation for dataset mode. Pure functions; GT never feeds the calibration itself.
import { angleBetweenDeg, fovFromFocal, matMul, matVec, reconstruct, rotationAngleDeg, transpose, umeyama } from "./stereo.js";

// jointMap: [layout index, GT joint index] pairs that denote the same anatomical point.
// MediaPipe Pose (33 landmarks) → Panoptic coco19.
const MP_TO_COCO19 = [
  [0, 1], [11, 3], [13, 4], [15, 5], [23, 6], [25, 7], [27, 8], [12, 9], [14, 10], [16, 11], [24, 12], [26, 13], [28, 14],
  [2, 15], [7, 16], [5, 17], [8, 18],
];
const COCO19_EDGES = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [2, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [1, 15], [15, 16], [1, 17], [17, 18]];

// MediaPipe hand (wrist, then each finger base → tip) → InterHand2.6M (each finger tip → base, wrist last; right 0–20,
// left 21–41). The "hands" layout keeps the right hand in slots 0–20 and the left in 21–41 as well.
const MP_HAND_TO_INTERHAND = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16];
const INTERHAND_PARENT = [1, 2, 3, 20, 5, 6, 7, 20, 9, 10, 11, 20, 13, 14, 15, 20, 17, 18, 19, 20, -1];

export const GT_SKELETONS = {
  coco19: { jointMap: MP_TO_COCO19, edges: COCO19_EDGES },
  interhand42: {
    jointMap: [...MP_HAND_TO_INTERHAND.map((g, k) => [k, g]), ...MP_HAND_TO_INTERHAND.map((g, k) => [21 + k, 21 + g])],
    edges: [0, 21].flatMap((o) => INTERHAND_PARENT.map((p, i) => [o + i, o + p]).filter(([, p]) => p >= o)),
  },
};

// Evaluation clips: dir holds clip/ (cam_<name>.mp4 + gt.json) and landmarks/ (headless detections).
export const DATASETS = {
  panoptic: { label: "CMU Panoptic — тело", dir: "data/panoptic-pose1", layout: "body", skeleton: "coco19" },
  interhand: { label: "InterHand2.6M — рука", dir: "data/interhand", layout: "hands", skeleton: "interhand42" },
};

// Bring GT to centimetres; joints are [x, y, z, validity/confidence] and only positive ones are scored.
export function normalizeGt(gt) {
  const k = { cm: 1, mm: 0.1 }[gt.units];
  if (!k) throw new Error(`unknown GT units: ${gt.units}`);
  return {
    ...gt,
    units: "cm",
    joints: gt.joints.map((f) => f && f.map(([x, y, z, c]) => [x * k, y * k, z * k, c])),
    cameras: Object.fromEntries(Object.entries(gt.cameras).map(([n, c]) => [n, { ...c, t: c.t.map((v) => v * k) }])),
  };
}

// Panoptic stores world→camera as X_c = R·X + t; express camera B relative to camera A.
export function gtRelativePose(gtA, gtB) {
  const R = matMul(gtB.R, transpose(gtA.R));
  const t = matVec(R, gtA.t).map((v, i) => gtB.t[i] - v);
  return { R, t, baseline: Math.hypot(...t) };
}

// GT camera pair in the same convention as calibratePair (A = world, |t| = 1), for the oracle run.
export function gtPairCameras(gtA, gtB) {
  const { R, t, baseline } = gtRelativePose(gtA, gtB);
  const cam = (g) => ({ f: (g.K[0][0] + g.K[1][1]) / 2, cx: g.K[0][2], cy: g.K[1][2] });
  return { camA: { ...cam(gtA), R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 0] }, camB: { ...cam(gtB), R, t: t.map((v) => v / baseline) } };
}

export function calibrationErrors(result, gtA, gtB) {
  const rel = gtRelativePose(gtA, gtB);
  return {
    fovA: { est: fovFromFocal(gtA.width, result.camA.f), gt: fovFromFocal(gtA.width, gtA.K[0][0]) },
    fovB: { est: fovFromFocal(gtB.width, result.camB.f), gt: fovFromFocal(gtB.width, gtB.K[0][0]) },
    angleBetweenCamerasDeg: rotationAngleDeg(rel.R, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    rotationErrorDeg: rotationAngleDeg(result.camB.R, rel.R),
    translationDirErrorDeg: angleBetweenDeg(result.camB.t, rel.t),
    baselineCm: rel.baseline,
  };
}

// Similarity alignment that ignores the worst 20% of points (refit twice), so a few wild triangulations
// cannot tilt the whole reconstruction onto GT.
export function robustSimilarity(src, dst) {
  let fit = umeyama(src, dst);
  for (let pass = 0; pass < 2; pass++) {
    const err = src.map((p, i) => Math.hypot(...fit.apply(p).map((v, k) => v - dst[i][k])));
    const cut = [...err].sort((a, b) => a - b)[Math.floor(err.length * 0.8)];
    const keep = src.map((_, i) => i).filter((i) => err[i] <= cut);
    fit = umeyama(keep.map((i) => src[i]), keep.map((i) => dst[i]));
  }
  return fit;
}

const stats = (values) => {
  const v = [...values].sort((a, b) => a - b);
  return { median: v[Math.floor(v.length / 2)], mean: v.reduce((s, x) => s + x, 0) / v.length, count: v.length };
};

// Pairs of (estimate, GT) joints for evaluated frames; estimate is null where the pipeline produced nothing.
function jointPairs(recon, gtJoints, jointMap) {
  return recon.map((frame, k) =>
    frame && gtJoints[k]
      ? jointMap.filter(([mp, g]) => frame[mp] && gtJoints[k][g][3] > 0).map(([mp, g]) => [frame[mp], gtJoints[k][g].slice(0, 3)])
      : [],
  );
}

// seq: one similarity transform for the whole clip — calibration + triangulation error with only the gauge removed.
// perFrame: Procrustes per frame (PA-MPJPE) — shape only, comparable with the monocular baseline.
export function poseErrors(recon, gtJoints, jointMap) {
  const pairs = jointPairs(recon, gtJoints, jointMap);
  const flat = pairs.flat();
  if (flat.length < 4) throw new Error(`only ${flat.length} reconstructed joints match valid GT joints`);
  const seqFit = robustSimilarity(flat.map(([X]) => X), flat.map(([, Y]) => Y));
  const seq = stats(flat.map(([X, Y]) => Math.hypot(...seqFit.apply(X).map((v, i) => v - Y[i]))));
  const frameErrors = pairs
    .filter((p) => p.length >= 8)
    .map((p) => {
      const fit = umeyama(p.map(([X]) => X), p.map(([, Y]) => Y));
      return p.reduce((s, [X, Y]) => s + Math.hypot(...fit.apply(X).map((v, i) => v - Y[i])), 0) / p.length;
    });
  return { seq, perFrame: stats(frameErrors), coverage: flat.length / gtJoints.reduce((n, f) => n + (f ? jointMap.filter(([, g]) => f[g][3] > 0).length : 0), 0), seqFit };
}

// Oracle: triangulate the same 2D detections with the GT cameras — the floor set by the 2D detector.
export function oracleReconstruction(frames, gtA, gtB) {
  return reconstruct(frames, gtPairCameras(gtA, gtB));
}

