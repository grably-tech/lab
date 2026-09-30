// Point layouts tracked per camera. Every layout is a fixed-length array of [u, v, visibility] so the stereo core stays
// generic; points that were not detected carry visibility 0. Pure functions — MediaPipe results are passed in.

// MediaPipe Pose limbs whose length stays constant.
export const BODY_BONES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [23, 24], [11, 23], [12, 24], [23, 25], [25, 27], [24, 26], [26, 28],
];

// MediaPipe Hands topology (0 wrist, 1–4 thumb, 5–8 index, 9–12 middle, 13–16 ring, 17–20 pinky). Every link is a
// phalanx or a palm edge, so all of them keep their length and double as calibration bones.
const HAND_LINKS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

export const RIGHT_HAND = 0;
export const LEFT_HAND = 21;
export const FACE = 42;

// Face Mesh points that sit on rigid bone and are not on the silhouette (the face oval shifts with the viewpoint):
// eye corners, nose bridge, nose tip, subnasale. All pairwise distances are constant.
export const FACE_MESH_POINTS = [33, 133, 362, 263, 168, 1, 2];
const FACE_LINKS = [[0, 1], [2, 3], [1, 4], [2, 4], [4, 5], [5, 6]];

// Expression contours as Face Mesh polylines (MediaPipe FACE_LANDMARKS_*): outer and inner lips, eyelids, brows, irises.
// They move with the facial expression, so they are triangulated but never used for calibration.
const EXPRESSION_LINES = [
  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291], [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308], [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  [263, 249, 390, 373, 374, 380, 381, 382, 362], [263, 466, 388, 387, 386, 385, 384, 398, 362],
  [33, 7, 163, 144, 145, 153, 154, 155, 133], [33, 246, 161, 160, 159, 158, 157, 173, 133],
  [276, 283, 282, 295, 285], [300, 293, 334, 296, 336], [46, 53, 52, 65, 55], [70, 63, 105, 66, 107],
  [474, 475, 476, 477, 474], [469, 470, 471, 472, 469],
];
const FACE_POINTS = [...new Set([...FACE_MESH_POINTS, ...EXPRESSION_LINES.flat()])];
const faceSlot = (m) => FACE + FACE_POINTS.indexOf(m);
const FACE_EDGES = EXPRESSION_LINES.flatMap((line) => line.slice(1).map((m, k) => [faceSlot(line[k]), faceSlot(m)]));

const shift = (links, offset) => links.map(([i, j]) => [i + offset, j + offset]);
const allPairs = (n, offset) => Array.from({ length: n }, (_, i) => Array.from({ length: n - i - 1 }, (_, k) => [offset + i, offset + i + k + 1])).flat();
const handsBones = [...shift(HAND_LINKS, RIGHT_HAND), ...shift(HAND_LINKS, LEFT_HAND)];

// groups: which detection a point belongs to — a mismatched or glitched detection is rejected as a whole.
// calibrationSize: points [0, calibrationSize) are rigid landmarks for calibration; the rest are only triangulated.
export const LAYOUTS = {
  body: {
    size: 33,
    calibrationSize: 33,
    groups: Array(33).fill("body"),
    bones: BODY_BONES,
    edges: [...BODY_BONES, [0, 2], [2, 7], [0, 5], [5, 8]],
  },
  hands: {
    size: 42,
    calibrationSize: 42,
    groups: [...Array(21).fill("right"), ...Array(21).fill("left")],
    bones: handsBones,
    edges: handsBones,
  },
  faceHands: {
    size: FACE + FACE_POINTS.length,
    calibrationSize: FACE + FACE_MESH_POINTS.length,
    groups: [...Array(21).fill("right"), ...Array(21).fill("left"), ...Array(FACE_POINTS.length).fill("face")],
    bones: [...handsBones, ...allPairs(FACE_MESH_POINTS.length, FACE)],
    edges: [...handsBones, ...shift(FACE_LINKS, FACE), ...FACE_EDGES],
  },
};

export const calibrationFrames = (frames, layout) =>
  frames.map((f) => ({ ...f, a: f.a.slice(0, layout.calibrationSize), b: f.b.slice(0, layout.calibrationSize) }));

// FaceLandmarker blendshapes (ARKit names) shown in live mode.
export const EXPRESSIONS = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "cheekPuff", "eyeBlinkLeft", "eyeBlinkRight", "browInnerUp"];

export function faceExpression(result) {
  const categories = result.faceBlendshapes[0]?.categories;
  if (!categories) return null;
  const score = new Map(categories.map((c) => [c.categoryName, c.score]));
  return Object.fromEntries(EXPRESSIONS.map((name) => [name, score.get(name)]));
}

// Mean over the cameras that see the face; null when none does.
export function mergeExpressions(expressions) {
  const seen = expressions.filter(Boolean);
  if (!seen.length) return null;
  return Object.fromEntries(EXPRESSIONS.map((name) => [name, seen.reduce((s, e) => s + e[name], 0) / seen.length]));
}

const inFrame = (p) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
export const emptyPoints = (size) => Array.from({ length: size }, () => [0, 0, 0]);

export function bodyPoints(landmarks, w, h) {
  return landmarks.map((p) => [p.x * w, p.y * h, inFrame(p) ? p.visibility ?? 0 : 0]);
}

// tasks-vision 1.0.1 labels the person's actual hand on raw (non-mirrored) frames — checked on InterHand2.6M, where
// the right hand came back as "Right" in every camera (older MediaPipe docs describe mirrored labels).
const HAND_SLOT = { Right: RIGHT_HAND, Left: LEFT_HAND };

// Fills hand slots in place from a HandLandmarker result. Two detections claiming the same hand are ambiguous, so
// that hand is left empty rather than guessed.
export function fillHands(points, result, w, h) {
  const bySlot = new Map();
  result.landmarks.forEach((landmarks, k) => {
    const slot = HAND_SLOT[result.handedness[k][0].categoryName];
    bySlot.set(slot, bySlot.has(slot) ? null : { landmarks, score: result.handedness[k][0].score });
  });
  for (const [slot, hand] of bySlot) {
    if (!hand) continue;
    hand.landmarks.forEach((p, j) => (points[slot + j] = [p.x * w, p.y * h, inFrame(p) ? hand.score : 0]));
  }
  return points;
}

export function fillFace(points, result, w, h) {
  const face = result.faceLandmarks[0];
  if (!face) return points;
  FACE_POINTS.forEach((m, k) => {
    const p = face[m];
    points[FACE + k] = [p.x * w, p.y * h, inFrame(p) ? 1 : 0];
  });
  return points;
}

// Distance between Face Mesh's outer eye corners (33 and 263): the one metric reference in the scene. Measured 8.3–8.4 cm
// on a CMU Panoptic subject triangulated with GT calibration — the mesh corners sit inside the anatomical ones (~9 cm
// average); people differ by about ±5%. Sets both the stereo scale and the single-camera face size.
export const OUTER_EYE_CORNERS_M = 0.084;

// Distance between the outer eye corners in a layout's 3D points; null unless both are there.
export const faceWidth = (points) => (points[FACE] && points[FACE + 3] ? Math.hypot(...points[FACE].map((v, i) => v - points[FACE + 3][i])) : null);

// Face mesh as a camera-aligned metric 3D shape: pixels (MediaPipe's z uses the x scale) scaled so the outer eye corners
// are an average distance apart — the same average-size footing as MediaPipe's metric hand landmarks.
export function fillFaceShape(shape, result, w, h) {
  const face = result.faceLandmarks[0];
  if (!face) return shape;
  const px = (m) => [face[m].x * w, face[m].y * h, face[m].z * w];
  const [a, b] = [px(33), px(263)];
  const k = OUTER_EYE_CORNERS_M / Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  FACE_POINTS.forEach((m, i) => (shape[FACE + i] = px(m).map((v) => v * k)));
  return shape;
}

// Hand-centred metric landmarks (the monocular baseline), in the same slots; null where absent. Each hand has its own
// origin, so this baseline is only comparable per hand.
export function handsWorld(result, size) {
  const world = Array(size).fill(null);
  const seen = new Set();
  result.worldLandmarks.forEach((landmarks, k) => {
    const slot = HAND_SLOT[result.handedness[k][0].categoryName];
    if (seen.has(slot)) {
      for (let j = 0; j < 21; j++) world[slot + j] = null;
      return;
    }
    seen.add(slot);
    landmarks.forEach((p, j) => (world[slot + j] = [p.x, p.y, p.z]));
  });
  return world;
}

export const hasAnyPoint = (points) => points.some((p) => p[2] > 0);
