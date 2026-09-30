import { describe, expect, test } from "bun:test";
import { DRIFT_FRAMES, drift, frameReprojectionPx, MIN_FRAMES, readiness } from "./autocal.js";
import { project, rotationFromVector } from "./stereo.js";

const size = { width: 1000, height: 1000 };
const camA = { f: 900, cx: 500, cy: 500, R: rotationFromVector([0, 0, 0]), t: [0, 0, 0] };
const camB = { f: 900, cx: 500, cy: 500, R: rotationFromVector([0, -0.3, 0]), t: [1, 0, 0.1] };
const frameAt = (points) => ({ a: points.map((X) => [...project(X, camA).slice(0, 2), 1]), b: points.map((X) => [...project(X, camB).slice(0, 2), 1]) });

describe("readiness", () => {
  const still = Array.from({ length: MIN_FRAMES }, () => frameAt([[0, 0, 3], [0.05, 0, 3], [0, 0.05, 3]]));
  const moving = Array.from({ length: MIN_FRAMES }, (_, k) => frameAt([[Math.sin(k / 10) * 0.8, Math.cos(k / 13) * 0.8, 3], [0.05, 0, 3]]));

  test("a still subject is not enough however long it is watched", () => {
    const r = readiness(still, size, size);
    expect(r.ready).toBe(false);
    expect(r.spread).toBeLessThan(0.05);
  });

  test("points moving across both images are, once there are enough frames", () => {
    expect(readiness(moving, size, size).ready).toBe(true);
    expect(readiness(moving.slice(0, MIN_FRAMES - 1), size, size).ready).toBe(false);
  });
});

describe("frameReprojectionPx", () => {
  test("exact projections of the calibrated cameras reproject with no error", () => {
    expect(frameReprojectionPx(frameAt([[0.1, 0.2, 3], [-0.3, 0.1, 2.5]]), { camA, camB })).toBeCloseTo(0, 6);
  });
});

describe("drift", () => {
  const calib = { medianReprojPx: 0.8, faceWidthUnits: 0.3 };
  const series = (v) => Array(DRIFT_FRAMES).fill(v);

  test("errors and face width as at calibration time: no drift", () => {
    expect(drift({ errors: series(1), widths: series(0.305) }, calib).drifted).toBe(false);
  });

  test("reprojection error above 3× the calibration's (at least 3 px) is a moved camera", () => {
    expect(drift({ errors: series(3.5), widths: [] }, calib)).toMatchObject({ reprojection: true, drifted: true });
  });

  test("a face that shrank by more than 8% is a moved camera", () => {
    expect(drift({ errors: series(1), widths: series(0.27) }, calib)).toMatchObject({ size: true, drifted: true });
  });

  test("too few frames yet: no verdict", () => {
    expect(drift({ errors: series(10).slice(1), widths: series(0.2).slice(1) }, calib).drifted).toBe(false);
  });

  test("a calibration without a face is judged by reprojection only", () => {
    expect(drift({ errors: series(1), widths: series(0.1) }, { medianReprojPx: 0.8, faceWidthUnits: null }).drifted).toBe(false);
  });
});
