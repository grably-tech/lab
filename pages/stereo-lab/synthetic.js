import { focalFromFov, matVec, mulberry32, project } from "./stereo.js";

// Kinematic tree with constant bone lengths (metres): [parent, child, length].
export const TREE = [
  [0, 1, 0.5], [1, 2, 0.25], [1, 3, 0.2], [3, 4, 0.3], [4, 5, 0.27], [1, 6, 0.2], [6, 7, 0.3], [7, 8, 0.27],
  [0, 9, 0.12], [9, 10, 0.45], [10, 11, 0.43], [0, 12, 0.12], [12, 13, 0.45], [13, 14, 0.43],
];
export const BONES = TREE.map(([a, b]) => [a, b]);

const unit = (v) => v.map((x) => x / Math.hypot(...v));

function syntheticMotion(frameCount, rand) {
  const smooth = () => {
    const phase = [rand(), rand(), rand()].map((x) => x * 6.28);
    const speed = [rand(), rand(), rand()].map((x) => 0.02 + 0.08 * x);
    return (k) => phase.map((p, i) => Math.sin(p + speed[i] * k));
  };
  const dirs = TREE.map(() => smooth());
  const root = smooth();
  return Array.from({ length: frameCount }, (_, k) => {
    const r = root(k);
    const joints = [[0.8 * r[0], 0.2 * r[1], 3 + 0.8 * r[2]]];
    TREE.forEach(([parent, child, length], bi) => {
      const d = unit(dirs[bi](k).map((x, i) => x + (i === 1 && child >= 9 ? 1.5 : 0) - (i === 1 && child === 1 ? 1.5 : 0)));
      joints[child] = joints[parent].map((x, i) => x + length * d[i]);
    });
    return joints;
  });
}

function lookAt(C, target) {
  const z = unit(target.map((x, i) => x - C[i]));
  const x = unit([z[2], 0, -z[0]]);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const R = [x, y, z];
  return { R, t: matVec(R, C).map((v) => -v) };
}

export function scenario({ angleDeg, fovA, fovB, noisePx, outlierRate, frames = 240, seed = 3 }) {
  const rand = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const sizeA = { width: 1280, height: 720 };
  const sizeB = { width: 1920, height: 1080 };
  const target = [0, 0, 3];
  const a = (angleDeg * Math.PI) / 180;
  const camA = { f: focalFromFov(sizeA.width, fovA), cx: 640, cy: 360, R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 0] };
  const camB = { f: focalFromFov(sizeB.width, fovB), cx: 960, cy: 540, ...lookAt([3 * Math.sin(a), -0.3, 3 - 3 * Math.cos(a)], target) };
  const motion = syntheticMotion(frames, rand);
  const observe = (X, cam, size) => {
    if (rand() < outlierRate) return [rand() * size.width, rand() * size.height, 0.9];
    const p = project(X, cam);
    return [p[0] + noisePx * gauss(), p[1] + noisePx * gauss(), 0.9];
  };
  // shapeA: exact 3D joints standing in for MediaPipe's monocular shape.
  const obs = motion.map((joints) => ({ a: joints.map((X) => observe(X, camA, sizeA)), b: joints.map((X) => observe(X, camB, sizeB)), shapeA: joints }));
  return { obs, motion, camA, camB, sizeA, sizeB };
}
