import { describe, expect, test } from "bun:test";
import { robustSimilarity } from "./eval.js";
import { BONES, scenario } from "./synthetic.js";
import {
  angleBetweenDeg,
  calibratePair,
  focalFromFov,
  fovFromFocal,
  matMul,
  matVec,
  mirrorCounts,
  mulberry32,
  reconstruct,
  rotationAngleDeg,
  rotationFromVector,
  svd3,
  transpose,
  umeyama,
  vectorFromRotation,
} from "./stereo.js";

describe("linear algebra", () => {
  test("svd3 reconstructs the matrix", () => {
    const A = [[2, -1, 0.5], [0.3, 4, 1], [-2, 0.1, 3]];
    const { U, S, V } = svd3(A);
    const B = matMul(matMul(U, [[S[0], 0, 0], [0, S[1], 0], [0, 0, S[2]]]), transpose(V));
    B.flat().forEach((v, i) => expect(v).toBeCloseTo(A.flat()[i], 9));
  });

  test("rotation vector round-trip, including near 180°", () => {
    for (const w of [[0.1, -0.2, 0.3], [0, 3.1, 0], [1.2, 1.2, -1.2]]) {
      expect(rotationAngleDeg(rotationFromVector(vectorFromRotation(rotationFromVector(w))), rotationFromVector(w))).toBeLessThan(1e-4);
    }
  });

  test("umeyama recovers a similarity transform", () => {
    const rand = mulberry32(5);
    const src = Array.from({ length: 20 }, () => [rand(), rand(), rand()]);
    const R = rotationFromVector([0.4, -1.1, 2.0]);
    const dst = src.map((p) => matVec(R, p).map((v, i) => 2.5 * v + [1, -2, 3][i]));
    const fit = umeyama(src, dst);
    expect(fit.s).toBeCloseTo(2.5, 9);
    expect(rotationAngleDeg(fit.R, R)).toBeLessThan(1e-6);
  });
});

describe("calibratePair on synthetic motion", () => {
  for (const angleDeg of [20, 50, 90]) {
    test(`${angleDeg}° between cameras, 2 px noise, 5% outliers`, () => {
      const s = scenario({ angleDeg, fovA: 70, fovB: 60, noisePx: 2, outlierRate: 0.05 });
      const res = calibratePair(s.obs, s.sizeA, s.sizeB, { bones: BONES, reprojSigmaPx: 2 });
      const fovAErr = fovFromFocal(1280, res.camA.f) - 70;
      const fovBErr = fovFromFocal(1920, res.camB.f) - 60;
      const rotErr = rotationAngleDeg(res.camB.R, s.camB.R);
      const tErr = angleBetweenDeg(res.camB.t, s.camB.t);

      const rec = reconstruct(s.obs, res);
      const pairs = rec.flatMap((f, k) => f.map((X, j) => [X, s.motion[k][j]])).filter(([X]) => X);
      const fit = robustSimilarity(pairs.map(([X]) => X), pairs.map(([, Y]) => Y));
      const errs = pairs.map(([X, Y]) => Math.hypot(...fit.apply(X).map((v, i) => v - Y[i]))).sort((x, y) => x - y);
      const medianMm = errs[Math.floor(errs.length / 2)] * 1000;

      console.log(`${angleDeg}°: ${res.log.join(" | ")} | ΔFOV ${fovAErr.toFixed(1)}/${fovBErr.toFixed(1)}° rot ${rotErr.toFixed(2)}° t ${tErr.toFixed(2)}° median3D ${medianMm.toFixed(0)} mm`);
      expect(Math.abs(fovAErr)).toBeLessThan(5);
      expect(Math.abs(fovBErr)).toBeLessThan(5);
      expect(rotErr).toBeLessThan(2);
      expect(tErr).toBeLessThan(5);
      expect(medianMm).toBeLessThan(50);
    }, 60000);
  }

  test("a remembered focal length is kept when it fits and corrected when it does not", () => {
    const s = scenario({ angleDeg: 50, fovA: 70, fovB: 60, noisePx: 2, outlierRate: 0.05 });
    const fov = (res) => [fovFromFocal(1280, res.camA.f), fovFromFocal(1920, res.camB.f)];
    const withPrior = (fovA, fovB) => calibratePair(s.obs, s.sizeA, s.sizeB, { bones: BONES, reprojSigmaPx: 2, focalPrior: { fA: focalFromFov(1280, fovA), fB: focalFromFov(1920, fovB) } });
    const right = fov(withPrior(70, 60));
    const wrong = fov(withPrior(45, 90));
    console.log(`prior 70/60 → ${right.map((v) => v.toFixed(1)).join("/")}, prior 45/90 → ${wrong.map((v) => v.toFixed(1)).join("/")}`);
    expect(Math.abs(right[0] - 70)).toBeLessThan(2);
    expect(Math.abs(right[1] - 60)).toBeLessThan(2);
    expect(Math.abs(wrong[0] - 70)).toBeLessThan(5);
    expect(Math.abs(wrong[1] - 60)).toBeLessThan(5);
  }, 60000);

  // Narrow cameras are nearly affine: the depth-reflected scene explains the images almost as well as the true one.
  test("narrow cameras: keeps the solution whose handedness matches the 3D shapes, not its depth-reflected twin", () => {
    const s = scenario({ angleDeg: 25, fovA: 15, fovB: 15, noisePx: 1, outlierRate: 0 });
    const res = calibratePair(s.obs, s.sizeA, s.sizeB, { bones: BONES, reprojSigmaPx: 2, fovRange: [5, 120] });
    const check = res.log.find((l) => l.startsWith("mirror check"));
    console.log(check);
    const { mirrored, total } = mirrorCounts(s.obs, reconstruct(s.obs, res), new Map([["all", [...Array(15).keys()]]])).all;
    expect(mirrored / total).toBeLessThan(0.2);
    expect(rotationAngleDeg(res.camB.R, s.camB.R)).toBeLessThan(10);
  }, 60000);
});
