import { describe, expect, test } from "bun:test";
import { calibrationFrames, emptyPoints, FACE, FACE_MESH_POINTS, faceWidth, fillFace, fillFaceShape, fillHands, LAYOUTS, OUTER_EYE_CORNERS_M, LEFT_HAND, mergeExpressions, RIGHT_HAND } from "./landmarks.js";

const hand = (x0) => Array.from({ length: 21 }, (_, j) => ({ x: x0 + j * 0.001, y: 0.5, z: 0 }));
const handResult = (...hands) => ({
  landmarks: hands.map(([, x0]) => hand(x0)),
  handedness: hands.map(([label, , score = 0.9]) => [{ categoryName: label, score }]),
});

describe("layouts", () => {
  test("every bone, edge and group refers to a point of its layout", () => {
    for (const layout of Object.values(LAYOUTS)) {
      expect(layout.groups).toHaveLength(layout.size);
      for (const [i, j] of layout.edges) {
        expect(i).toBeLessThan(layout.size);
        expect(j).toBeLessThan(layout.size);
      }
      for (const [i, j] of layout.bones) {
        expect(i).toBeLessThan(layout.calibrationSize);
        expect(j).toBeLessThan(layout.calibrationSize);
      }
    }
  });

  test("calibration sees only the rigid points, expression contours stay out", () => {
    const { size, calibrationSize } = LAYOUTS.faceHands;
    expect(calibrationSize).toBe(FACE + FACE_MESH_POINTS.length);
    expect(size).toBeGreaterThan(calibrationSize);
    const [frame] = calibrationFrames([{ a: emptyPoints(size), b: emptyPoints(size), t: 5 }], LAYOUTS.faceHands);
    expect(frame.a).toHaveLength(calibrationSize);
    expect(frame.b).toHaveLength(calibrationSize);
    expect(frame.t).toBe(5);
  });
});

describe("fillHands", () => {
  test("puts each labelled hand into its own slot, in pixels, with the handedness score as visibility", () => {
    const points = fillHands(emptyPoints(42), handResult(["Right", 0.2, 0.95], ["Left", 0.6, 0.8]), 1000, 500);
    expect(points[RIGHT_HAND]).toEqual([200, 250, 0.95]);
    expect(points[LEFT_HAND][0]).toBeCloseTo(600);
    expect(points[LEFT_HAND][2]).toBe(0.8);
  });

  test("two detections claiming the same hand leave that hand empty instead of guessing", () => {
    const points = fillHands(emptyPoints(42), handResult(["Right", 0.2], ["Right", 0.6]), 1000, 500);
    expect(points.every((p) => p[2] === 0)).toBe(true);
  });

  test("points outside the frame are not observations", () => {
    const points = fillHands(emptyPoints(42), handResult(["Left", 0.999]), 100, 100);
    expect(points[LEFT_HAND][2]).toBeGreaterThan(0);
    expect(points[LEFT_HAND + 20][2]).toBe(0);
  });
});

describe("fillFace", () => {
  test("takes the rigid Face Mesh points first, then the expression contours", () => {
    const face = Array.from({ length: 478 }, (_, m) => ({ x: m / 1000, y: 0.25, z: 0 }));
    const points = fillFace(emptyPoints(LAYOUTS.faceHands.size), { faceLandmarks: [face] }, 1000, 400);
    FACE_MESH_POINTS.forEach((m, k) => expect(points[FACE + k]).toEqual([m, 100, 1]));
    expect(points.slice(FACE).every((p) => p[2] === 1)).toBe(true);
    expect(new Set(points.slice(FACE).map((p) => p[0])).size).toBe(LAYOUTS.faceHands.size - FACE);
  });

  test("face shape keeps MediaPipe's relative depth on the x pixel scale, scaled to the reference outer eye corner distance", () => {
    const face = Array.from({ length: 478 }, (_, m) => ({ x: 0.5, y: 0.25, z: m / 1000 }));
    const shape = fillFaceShape(Array(LAYOUTS.faceHands.size).fill(null), { faceLandmarks: [face] }, 1000, 400);
    const k = OUTER_EYE_CORNERS_M / (263 - 33);
    [500, 100, FACE_MESH_POINTS[4]].forEach((v, i) => expect(shape[FACE + 4][i]).toBeCloseTo(v * k, 12));
    expect(shape.slice(0, FACE).every((p) => p === null)).toBe(true);
  });

  test("face width is the distance between the outer eye corners (Face Mesh 33 and 263)", () => {
    expect([FACE_MESH_POINTS[0], FACE_MESH_POINTS[3]]).toEqual([33, 263]);
    const points = Array(LAYOUTS.faceHands.size).fill(null);
    expect(faceWidth(points)).toBeNull();
    points[FACE] = [0, 0, 1];
    points[FACE + 3] = [3, 4, 1];
    expect(faceWidth(points)).toBe(5);
  });

  test("no face leaves the slots empty", () => {
    const points = fillFace(emptyPoints(LAYOUTS.faceHands.size), { faceLandmarks: [] }, 1000, 400);
    expect(points.every((p) => p[2] === 0)).toBe(true);
  });
});

describe("mergeExpressions", () => {
  const expression = (v) => ({ jawOpen: v, mouthSmileLeft: v, mouthSmileRight: v, mouthPucker: v, cheekPuff: v, eyeBlinkLeft: v, eyeBlinkRight: v, browInnerUp: v });

  test("averages the cameras that see the face", () => {
    expect(mergeExpressions([expression(0.2), expression(0.6)]).jawOpen).toBeCloseTo(0.4);
    expect(mergeExpressions([null, expression(0.6)]).jawOpen).toBe(0.6);
  });

  test("no face in any camera gives no expression", () => {
    expect(mergeExpressions([null, null])).toBeNull();
  });
});
