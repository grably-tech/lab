import { describe, expect, test } from "bun:test";
import { depthJitter, depthOffset, fuseFrame, groupDepths, groupsOf, monocularFrame, placeShape, stereoDepths } from "./mono.js";
import { matVec, median, project, reconstruct, rotationFromVector } from "./stereo.js";

const camA = { f: 900, cx: 640, cy: 360, R: rotationFromVector([0, 0, 0]), t: [0, 0, 0] };
const camB = { f: 1100, cx: 640, cy: 360, R: rotationFromVector([0.05, -0.6, 0.02]), t: [1, 0.05, 0.2] };
const layout = { groups: ["hand", "hand", "hand", "hand", "face", "face", "face"] };
const groups = groupsOf(layout);
const world = [[0.1, 0.2, 3], [0.3, 0.1, 3.2], [0.2, 0.45, 2.9], [0.05, 0.3, 3.1], [-0.4, -0.3, 3.5], [-0.3, -0.35, 3.45], [-0.35, -0.2, 3.4]];
const observe = (cam) => world.map((X) => [...project(X, cam).slice(0, 2), 1]);
// MediaPipe-like shape: camera-aligned axes, own origin and unit.
const shapeIn = (cam) => world.map((X) => matVec(cam.R, X).map((v, i) => (v + cam.t[i] - 5) * 100));
const inCamera = (X, cam) => matVec(cam.R, X).map((v, i) => v + cam.t[i]);
const hidden = world.map(() => [0, 0, 0]);

describe("placeShape", () => {
  test("recovers the points from one camera given the true depth", () => {
    const slots = groups.get("hand");
    const depth = median(slots.map((j) => inCamera(world[j], camB)[2]));
    const placed = placeShape(observe(camB), shapeIn(camB), slots, depth, camB);
    for (const j of slots) world[j].forEach((v, i) => expect(placed.get(j)[i]).toBeCloseTo(v, 9));
  });
});

describe("fuseFrame", () => {
  const frame = { a: observe(camA), b: observe(camB), shapeA: shapeIn(camA), shapeB: shapeIn(camB) };
  const recon = reconstruct([frame], { camA, camB })[0];
  const anchors = stereoDepths(frame, recon, groups, { camA, camB });

  test("uses stereo where both cameras see a group and places the rest from the camera that does", () => {
    const { points, source } = fuseFrame({ ...frame, b: frame.b.map((p, j) => (layout.groups[j] === "face" ? [0, 0, 0] : p)) }, recon, { camA, camB }, anchors, groups);
    expect(source).toEqual({ hand: "stereo", face: "A" });
    points.forEach((X, j) => X.forEach((v, i) => expect(v).toBeCloseTo(world[j][i], 6)));
  });

  test("a group never seen in stereo has no depth anchor and is not shown", () => {
    const { points, source } = fuseFrame({ ...frame, b: hidden }, recon, { camA, camB }, {}, groups);
    expect(source).toEqual({});
    expect(points.every((X) => X === null)).toBe(true);
  });
});

describe("monocularFrame", () => {
  test("a shape with the true size lands on the true points: depth comes from apparent size", () => {
    const metricShape = world.map((X) => inCamera(X, camA).map((v, i) => v - [0.1, -0.2, 3][i]));
    const points = monocularFrame(observe(camA), metricShape, groups, camA);
    points.forEach((X, j) => X.forEach((v, i) => expect(v).toBeCloseTo(world[j][i], 9)));
  });

  test("a shape 10% too big puts the group 10% farther", () => {
    const bigShape = world.map((X) => inCamera(X, camA).map((v) => v * 1.1));
    const depths = groupDepths(monocularFrame(observe(camA), bigShape, groups, camA), groups);
    const trueDepths = groupDepths(world, groups);
    for (const g of ["hand", "face"]) expect(depths[g] / trueDepths[g]).toBeCloseTo(1.1, 9);
  });
});

describe("depthJitter", () => {
  test("median jump between consecutive frames, skipping frames where the group is missing", () => {
    expect(depthJitter([{ hand: 1 }, { hand: 1.5 }, {}, { hand: 9 }, { hand: 8.9 }, { hand: 9.1 }])).toEqual({ hand: expect.closeTo(0.2, 9) });
  });
});

describe("depthOffset", () => {
  test("median difference in common units over frames where both histories have the group", () => {
    const mono = [{ hand: 0.5 }, { hand: 0.55 }, { hand: 0.6 }, {}, { hand: 1 }];
    const stereo = [{ hand: 2 }, { hand: 2.2 }, { hand: 2.2 }, { hand: 2.1 }, {}];
    expect(depthOffset(mono, stereo, 100, 20)).toEqual({ hand: expect.closeTo(11, 9) });
  });
});
