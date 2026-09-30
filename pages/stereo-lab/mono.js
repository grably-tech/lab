// Single-camera placement for a detection (hand, face) that only one camera sees. The camera gives the 2D points and
// MediaPipe gives a 3D shape in that camera's orientation, but depth from apparent size is useless: on InterHand2.6M it
// jumps by ~10 cm between frames 0.2 s apart. So depth is held from the last frame both cameras saw the detection
// (2–2.5 cm error after up to 5 s on InterHand, where the hand stays at a similar depth). Pure functions.
import { matVec, median, project, solveLinear, transpose } from "./stereo.js";

const MIN_VISIBILITY = 0.5;

// Detection group name → its point slots.
export function groupsOf(layout) {
  const groups = new Map();
  layout.groups.forEach((g, i) => groups.set(g, [...(groups.get(g) ?? []), i]));
  return groups;
}

const detected = (image, slots) => slots.some((j) => image[j][2] >= MIN_VISIBILITY);

// Translates shape (camera-aligned axes, any origin) so it reprojects onto the 2D detections of cam — linear least
// squares in the translation. The shape's own scale sets the depth. Returns camera-frame points by slot.
function fitShape(image, shape, slots, cam) {
  const AtA = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const Atb = [0, 0, 0];
  const addRow = (row, rhs) =>
    row.forEach((v, i) => {
      row.forEach((w, k) => (AtA[i][k] += v * w));
      Atb[i] += v * rhs;
    });
  for (const j of slots) {
    if (!shape[j] || image[j][2] < MIN_VISIBILITY) continue;
    const [X, Y, Z] = shape[j];
    const du = image[j][0] - cam.cx;
    const dv = image[j][1] - cam.cy;
    addRow([cam.f, 0, -du], du * Z - cam.f * X);
    addRow([0, cam.f, -dv], dv * Z - cam.f * Y);
  }
  const t = solveLinear(AtA, Atb);
  return new Map(slots.filter((j) => shape[j]).map((j) => [j, shape[j].map((v, i) => v + t[i])]));
}

// Camera-frame points, scaled by k about the camera centre, to world (camera A frame).
const toWorld = (inCamera, cam, k = 1) => {
  const Rt = transpose(cam.R);
  return new Map([...inCamera].map(([j, X]) => [j, matVec(Rt, X.map((v, i) => k * v - cam.t[i]))]));
};

// The shape fitted to the detections, then scaled about the camera centre to the given median depth. World points.
export function placeShape(image, shape, slots, depth, cam) {
  const inCamera = fitShape(image, shape, slots, cam);
  return toWorld(inCamera, cam, depth / median([...inCamera.values()].map((X) => X[2])));
}

// What one camera alone gives: every detected group's metric MediaPipe shape (average hand/face size) fitted to the
// detections, so depth comes only from apparent size. World points (camera A frame) in metres; null where absent.
export function monocularFrame(image, shape, groups, cam) {
  const points = image.map(() => null);
  for (const [, slots] of groups) {
    if (!slots.some((j) => shape[j] && image[j][2] >= MIN_VISIBILITY)) continue;
    for (const [j, X] of toWorld(fitShape(image, shape, slots, cam), cam)) points[j] = X;
  }
  return points;
}

// Median depth (camera A frame, z) of each group that has points.
export function groupDepths(points, groups) {
  const depths = {};
  for (const [name, slots] of groups) {
    const zs = slots.filter((j) => points[j]).map((j) => points[j][2]);
    if (zs.length) depths[name] = median(zs);
  }
  return depths;
}

// Depth jitter per group: median absolute change of its depth between consecutive frames of a history of
// groupDepths results. Only consecutive frames that both have the group count.
export function depthJitter(history) {
  const steps = {};
  for (let k = 1; k < history.length; k++) {
    for (const [name, z] of Object.entries(history[k])) if (name in history[k - 1]) (steps[name] ??= []).push(Math.abs(z - history[k - 1][name]));
  }
  return Object.fromEntries(Object.entries(steps).map(([name, v]) => [name, median(v)]));
}

// Median depth of each group in both cameras' frames, for groups both cameras detected — the anchors for later
// single-camera frames.
export function stereoDepths(frame, recon, groups, { camA, camB }) {
  const depths = {};
  for (const [name, slots] of groups) {
    const points = slots.map((j) => recon[j]).filter(Boolean);
    if (!points.length || !detected(frame.a, slots) || !detected(frame.b, slots)) continue;
    depths[name] = { A: median(points.map((X) => project(X, camA)[2])), B: median(points.map((X) => project(X, camB)[2])) };
  }
  return depths;
}

// Per group: stereo points when both cameras detected it; otherwise placed from the one camera that did, at the depth
// anchored by an earlier stereo frame. source[group] is "stereo", "A", "B" or absent (not shown).
export function fuseFrame(frame, recon, { camA, camB }, anchors, groups) {
  const points = recon.map(() => null);
  const source = {};
  for (const [name, slots] of groups) {
    const inA = detected(frame.a, slots);
    const inB = detected(frame.b, slots);
    if (inA && inB) {
      for (const j of slots) points[j] = recon[j];
      source[name] = "stereo";
      continue;
    }
    const view = inA ? { label: "A", image: frame.a, shape: frame.shapeA, cam: camA } : inB ? { label: "B", image: frame.b, shape: frame.shapeB, cam: camB } : null;
    if (!view || !anchors[name] || !slots.some((j) => view.shape[j] && view.image[j][2] >= MIN_VISIBILITY)) continue;
    for (const [j, X] of placeShape(view.image, view.shape, slots, anchors[name][view.label], view.cam)) points[j] = X;
    source[name] = view.label;
  }
  return { points, source };
}

// Systematic depth offset per group between two aligned histories of groupDepths (a − b, each scaled to common units):
// the median over frames where both have the group.
export function depthOffset(historyA, historyB, scaleA, scaleB) {
  const diffs = {};
  historyA.forEach((a, k) => {
    for (const [name, z] of Object.entries(a)) if (name in historyB[k]) (diffs[name] ??= []).push(z * scaleA - historyB[k][name] * scaleB);
  });
  return Object.fromEntries(Object.entries(diffs).map(([name, v]) => [name, median(v)]));
}
