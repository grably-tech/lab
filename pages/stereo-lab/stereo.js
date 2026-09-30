// Calibration-free two-view reconstruction of a moving person from 2D landmarks.
// Pure functions only: the browser page and the offline evaluation share this module.
//
// Pipeline: joint correspondences → fundamental matrix (normalized 8-point + RANSAC) →
// focal lengths (1-D valley of the essential constraint, disambiguated by bone-length constancy) →
// relative pose (E decomposition + cheirality) → Levenberg–Marquardt refinement of
// reprojection error plus bone-length constancy (the person is the calibration object).
// Camera A is the world frame: A = K_A[I|0], B = K_B[R|t], |t| = 1 (scale is unobservable).

// ---------- small dense linear algebra (row-major arrays) ----------

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
export const matMul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
export const matVec = (A, v) => A.map((row) => row.reduce((s, x, k) => s + x * v[k], 0));
const identity = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const det3 = (M) =>
  M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
  M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
  M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

// Symmetric eigen-decomposition by cyclic Jacobi. Eigenvalues ascending; vectors[i] pairs with values[i].
export function eigSym(M) {
  const n = M.length;
  const a = M.map((row) => row.slice());
  const v = identity(n);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        // a ← Jᵀ a J with J = I except J[p][p]=J[q][q]=c, J[p][q]=s, J[q][p]=−s.
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => a[i][i] - a[j][j]);
  return { values: order.map((i) => a[i][i]), vectors: order.map((i) => v.map((row) => row[i])) };
}

// 3×3 SVD via eig(AᵀA): A = U·diag(S)·Vᵀ, S descending; U is a proper rotation, so S[2] may be negative.
export function svd3(A) {
  const { values, vectors } = eigSym(matMul(transpose(A), A));
  const vs = [vectors[2], vectors[1], vectors[0]];
  const s0 = Math.sqrt(Math.max(values[2], 0));
  const s1 = Math.sqrt(Math.max(values[1], 0));
  const u0 = matVec(A, vs[0]).map((x) => x / s0);
  let u1 = matVec(A, vs[1]).map((x) => x / (s1 || 1));
  u1 = sub3(u1, u0.map((x) => x * dot3(u0, u1)));
  u1 = u1.map((x) => x / norm3(u1));
  const u2 = cross3(u0, u1);
  const s2 = dot3(u2, matVec(A, vs[2]));
  return { U: transpose([u0, u1, u2]), S: [s0, s1, s2], V: transpose(vs) };
}

// Gaussian elimination with partial pivoting for small dense systems.
export function solveLinear(A, b) {
  const n = b.length;
  const m = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[pivot][c])) pivot = r;
    [m[c], m[pivot]] = [m[pivot], m[c]];
    if (Math.abs(m[c][c]) < 1e-300) throw new Error("singular system");
    for (let r = c + 1; r < n; r++) {
      const k = m[r][c] / m[c][c];
      for (let j = c; j <= n; j++) m[r][j] -= k * m[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r][n];
    for (let j = r + 1; j < n; j++) s -= m[r][j] * x[j];
    x[r] = s / m[r][r];
  }
  return x;
}

// ---------- rotations ----------

export function rotationFromVector(w) {
  const theta = norm3(w);
  if (theta < 1e-12) return [[1, -w[2], w[1]], [w[2], 1, -w[0]], [-w[1], w[0], 1]];
  const [x, y, z] = w.map((c) => c / theta);
  const s = Math.sin(theta);
  const c1 = 1 - Math.cos(theta);
  const K = [[0, -z, y], [z, 0, -x], [-y, x, 0]];
  const K2 = matMul(K, K);
  return identity(3).map((row, i) => row.map((v, j) => v + s * K[i][j] + c1 * K2[i][j]));
}

export function vectorFromRotation(R) {
  const cosT = Math.min(1, Math.max(-1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2));
  const theta = Math.acos(cosT);
  const axisRaw = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]];
  if (theta < 1e-9) return [0, 0, 0];
  if (Math.PI - theta > 1e-4) return axisRaw.map((v) => (v * theta) / (2 * Math.sin(theta)));
  // Near 180°: R + I = 2·axis·axisᵀ; take its largest column.
  const B = R.map((row, i) => row.map((v, j) => (v + (i === j ? 1 : 0)) / 2));
  const col = [0, 1, 2].reduce((best, j) => (B[j][j] > B[best][best] ? j : best), 0);
  const axis = [B[0][col], B[1][col], B[2][col]];
  const n = norm3(axis);
  return axis.map((v) => (v / n) * theta);
}

export const rotationAngleDeg = (R1, R2) => {
  const R = matMul(R1, transpose(R2));
  return (Math.acos(Math.min(1, Math.max(-1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))) * 180) / Math.PI;
};

export const angleBetweenDeg = (a, b) => (Math.acos(Math.min(1, Math.max(-1, dot3(a, b) / (norm3(a) * norm3(b))))) * 180) / Math.PI;

// ---------- camera model: pinhole, square pixels, principal point at the image centre ----------

export const focalFromFov = (width, hfovDeg) => width / 2 / Math.tan((hfovDeg * Math.PI) / 360);
export const fovFromFocal = (width, f) => (2 * Math.atan(width / 2 / f) * 180) / Math.PI;

const toNormalized = (p, cam) => [(p[0] - cam.cx) / cam.f, (p[1] - cam.cy) / cam.f];

export function project(X, cam) {
  const Xc = [dot3(cam.R[0], X) + cam.t[0], dot3(cam.R[1], X) + cam.t[1], dot3(cam.R[2], X) + cam.t[2]];
  return [(cam.f * Xc[0]) / Xc[2] + cam.cx, (cam.f * Xc[1]) / Xc[2] + cam.cy, Xc[2]];
}

// Linear triangulation from ≥2 views in normalized coordinates (inhomogeneous least squares).
export function triangulate(observations) {
  const N = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b = [0, 0, 0];
  for (const { p, cam } of observations) {
    const [x, y] = toNormalized(p, cam);
    const { R, t } = cam;
    const rows = [
      [x * R[2][0] - R[0][0], x * R[2][1] - R[0][1], x * R[2][2] - R[0][2], x * t[2] - t[0]],
      [y * R[2][0] - R[1][0], y * R[2][1] - R[1][1], y * R[2][2] - R[1][2], y * t[2] - t[1]],
    ];
    for (const r of rows) {
      for (let i = 0; i < 3; i++) {
        b[i] -= r[i] * r[3];
        for (let j = 0; j < 3; j++) N[i][j] += r[i] * r[j];
      }
    }
  }
  return solveLinear(N, b);
}

// ---------- fundamental matrix ----------

function hartley(points) {
  const cx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const cy = points.reduce((s, p) => s + p[1], 0) / points.length;
  const d = points.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / points.length;
  const k = Math.SQRT2 / d;
  return [[k, 0, -k * cx], [0, k, -k * cy], [0, 0, 1]];
}

export function fundamental8(p1, p2) {
  const T1 = hartley(p1);
  const T2 = hartley(p2);
  const ATA = Array.from({ length: 9 }, () => new Array(9).fill(0));
  for (let i = 0; i < p1.length; i++) {
    const [u1, v1] = matVec(T1, [p1[i][0], p1[i][1], 1]);
    const [u2, v2] = matVec(T2, [p2[i][0], p2[i][1], 1]);
    const row = [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1];
    for (let a = 0; a < 9; a++) for (let b = 0; b < 9; b++) ATA[a][b] += row[a] * row[b];
  }
  const f = eigSym(ATA).vectors[0];
  const { U, S, V } = svd3([f.slice(0, 3), f.slice(3, 6), f.slice(6, 9)]);
  const Fn = matMul(matMul(U, [[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]]), transpose(V));
  const F = matMul(matMul(transpose(T2), Fn), T1);
  const n = Math.hypot(...F.flat());
  return F.map((row) => row.map((v) => v / n));
}

export function sampsonError(F, a, b) {
  const x1 = [a[0], a[1], 1];
  const x2 = [b[0], b[1], 1];
  const Fx1 = matVec(F, x1);
  const Ftx2 = matVec(transpose(F), x2);
  const e = dot3(x2, Fx1);
  return (e * e) / (Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2);
}

// Deterministic PRNG so runs are reproducible.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function ransacFundamental(p1, p2, { thresholdPx, iterations = 800, seed = 1 }) {
  const rand = mulberry32(seed);
  const thr2 = thresholdPx * thresholdPx;
  let best = { inliers: [] };
  for (let it = 0; it < iterations; it++) {
    const sample = new Set();
    while (sample.size < 8) sample.add(Math.floor(rand() * p1.length));
    const idx = [...sample];
    const F = fundamental8(idx.map((i) => p1[i]), idx.map((i) => p2[i]));
    const inliers = [];
    for (let i = 0; i < p1.length; i++) if (sampsonError(F, p1[i], p2[i]) < thr2) inliers.push(i);
    if (inliers.length > best.inliers.length) best = { F, inliers };
  }
  const F = fundamental8(best.inliers.map((i) => p1[i]), best.inliers.map((i) => p2[i]));
  const inliers = [];
  for (let i = 0; i < p1.length; i++) if (sampsonError(F, p1[i], p2[i]) < thr2) inliers.push(i);
  return { F, inliers };
}

// ---------- focal lengths and relative pose from F ----------

const intrinsics = (f, cx, cy) => [[f, 0, cx], [0, f, cy], [0, 0, 1]];

// E = K_Bᵀ F K_A must have two equal singular values; score how far it is from that.
export const essentialCost = (F, camA, camB) => {
  const E = matMul(matMul(transpose(intrinsics(camB.f, camB.cx, camB.cy)), F), intrinsics(camA.f, camA.cx, camA.cy));
  const { S } = svd3(E);
  return (S[0] - S[1]) / S[0];
};

// Cameras aimed at the same person have (nearly) intersecting optical axes — the classic degenerate case
// where F fixes focals only up to a one-parameter family. So return that family: for every FOV_A the FOV_B
// that best satisfies the essential constraint. The body picks the true member later.
export function focalValley(F, sizeA, sizeB, { fovRange, steps = 48 }) {
  const fovs = Array.from({ length: steps }, (_, i) => fovRange[0] + ((fovRange[1] - fovRange[0]) * i) / (steps - 1));
  const center = (s) => ({ cx: s.width / 2, cy: s.height / 2 });
  return fovs.map((fovA) => {
    const camA = { f: focalFromFov(sizeA.width, fovA), ...center(sizeA) };
    let best = { cost: Infinity };
    for (const fovB of fovs) {
      const camB = { f: focalFromFov(sizeB.width, fovB), ...center(sizeB) };
      const cost = essentialCost(F, camA, camB);
      if (cost < best.cost) best = { cost, intrA: camA, intrB: camB };
    }
    return best;
  });
}

export const median = (values) => {
  const v = [...values].sort((x, y) => x - y);
  return v[Math.floor(v.length / 2)];
};

// tracks: [{ bone: [i, j], frames: [k, …] }] — each bone only over frames where both its ends are observed.
function trackLengths(frames, { bone: [i, j], frames: ks }, camA, camB) {
  const point = (k, joint) => triangulate([{ p: frames[k].a[joint], cam: camA }, { p: frames[k].b[joint], cam: camB }]);
  return ks.map((k) => norm3(sub3(point(k, i), point(k, j))));
}

// How much bone lengths fluctuate across frames: median over bones of MAD/median. A wrong focal/pose
// reconstructs the body with a projective warp, so limbs stretch as the person moves through the volume.
export function boneVariation(frames, tracks, camA, camB) {
  return median(
    tracks.map((track) => {
      const lengths = trackLengths(frames, track, camA, camB);
      const m = median(lengths);
      return median(lengths.map((x) => Math.abs(x - m))) / m;
    }),
  );
}

// Four (R, t) candidates from E; keep the one that puts the most points in front of both cameras.
export function poseFromEssential(E, pairs, camAIntr, camBIntr) {
  let { U, V } = svd3(E);
  if (det3(U) < 0) U = U.map((row) => [row[0], row[1], -row[2]]);
  if (det3(V) < 0) V = V.map((row) => [row[0], row[1], -row[2]]);
  const W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
  const Rs = [matMul(matMul(U, W), transpose(V)), matMul(matMul(U, transpose(W)), transpose(V))];
  const u3 = [U[0][2], U[1][2], U[2][2]];
  let best = { score: -1 };
  for (const R of Rs) {
    for (const t of [u3, u3.map((v) => -v)]) {
      const camA = { ...camAIntr, R: identity(3), t: [0, 0, 0] };
      const camB = { ...camBIntr, R, t };
      let score = 0;
      for (const [a, b] of pairs) {
        const X = triangulate([{ p: a, cam: camA }, { p: b, cam: camB }]);
        if (X[2] > 0 && project(X, camB)[2] > 0) score++;
      }
      if (score > best.score) best = { score, R, t };
    }
  }
  return best;
}

// ---------- nonlinear refinement ----------

export function levenbergMarquardt(residualFn, p0, { maxIterations = 60, tolerance = 1e-9 } = {}) {
  let p = p0.slice();
  let r = residualFn(p);
  let cost = r.reduce((s, v) => s + v * v, 0);
  let lambda = 1e-3;
  for (let iter = 0; iter < maxIterations; iter++) {
    const J = p.map((pj, j) => {
      const h = 1e-6 * Math.max(1, Math.abs(pj));
      const q = p.slice();
      q[j] += h;
      const rq = residualFn(q);
      return rq.map((v, i) => (v - r[i]) / h);
    });
    const n = p.length;
    const JTJ = Array.from({ length: n }, (_, a) => Array.from({ length: n }, (_, b) => J[a].reduce((s, v, i) => s + v * J[b][i], 0)));
    const JTr = J.map((col) => col.reduce((s, v, i) => s + v * r[i], 0));
    let improved = false;
    for (let attempt = 0; attempt < 10; attempt++) {
      const A = JTJ.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lambda) + 1e-12 : v)));
      const step = solveLinear(A, JTr.map((v) => -v));
      const q = p.map((v, i) => v + step[i]);
      const rq = residualFn(q);
      const costQ = rq.reduce((s, v) => s + v * v, 0);
      if (Number.isFinite(costQ) && costQ < cost) {
        const relative = (cost - costQ) / cost;
        p = q;
        r = rq;
        cost = costQ;
        lambda = Math.max(lambda / 10, 1e-12);
        improved = relative > tolerance;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return { params: p, cost };
}

// Huber loss written as a residual so that Σr² equals the robust cost.
const huber = (r, delta) => (Math.abs(r) <= delta ? r : Math.sign(r) * Math.sqrt(2 * delta * Math.abs(r) - delta * delta));
// Cauchy loss as a residual: grows only logarithmically, so wild samples (a joint that slid along the epipolar
// line and triangulated far away) cannot drag the geometry the way they can under Huber's linear tail.
const cauchy = (r, c) => Math.sign(r) * c * Math.sqrt(Math.log1p((r / c) ** 2));

const unitFromAngles = (theta, phi) => [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
const anglesFromUnit = (t) => [Math.acos(Math.max(-1, Math.min(1, t[2] / norm3(t)))), Math.atan2(t[1], t[0])];

function camerasFromParams(p, sizeA, sizeB) {
  const [logFA, logFB, wx, wy, wz, theta, phi] = p;
  return {
    camA: { f: Math.exp(logFA), cx: sizeA.width / 2, cy: sizeA.height / 2, R: identity(3), t: [0, 0, 0] },
    camB: { f: Math.exp(logFB), cx: sizeB.width / 2, cy: sizeB.height / 2, R: rotationFromVector([wx, wy, wz]), t: unitFromAngles(theta, phi) },
  };
}

// ---------- public API ----------

// frames[k] = { a: [[u, v, visibility] × J], b: [[u, v, visibility] × J] } in pixels, time-synchronised.
export function collectCorrespondences(frames, minVisibility) {
  const list = [];
  frames.forEach((frame, k) => {
    frame.a.forEach((pa, j) => {
      const pb = frame.b[j];
      if (pa[2] >= minVisibility && pb[2] >= minVisibility) list.push({ frame: k, joint: j, a: pa, b: pb });
    });
  });
  return list;
}

const rmsAfter = (fit, src, dst) => Math.sqrt(src.reduce((s, p, i) => s + fit.apply(p).reduce((q, v, k) => q + (v - dst[i][k]) ** 2, 0), 0) / src.length);

// Per group: how many stereo observations fit MediaPipe's own 3D shape of the group (frame.shapeA, camera A) better
// mirrored than as is. MediaPipe shapes keep true handedness, so a reconstruction reflected in depth shows up as
// mirrored ≈ total (InterHand2.6M: 0–11% on correct reconstructions, 89–100% on reflected ones).
export function mirrorCounts(frames, recon, groupSlots) {
  const counts = {};
  frames.forEach((frame, k) => {
    for (const [name, slots] of groupSlots) {
      const pairs = slots.filter((j) => recon[k][j] && frame.shapeA[j]).map((j) => [frame.shapeA[j], recon[k][j]]);
      if (pairs.length < 6) continue;
      const src = pairs.map(([s]) => s);
      const dst = pairs.map(([, X]) => X);
      const flipped = src.map(([x, y, z]) => [-x, y, z]);
      counts[name] ??= { mirrored: 0, total: 0 };
      counts[name].total++;
      if (rmsAfter(umeyama(flipped, dst), flipped, dst) < rmsAfter(umeyama(src, dst), src, dst)) counts[name].mirrored++;
    }
  });
  return counts;
}

// The two-view Necker ambiguity: under weak perspective, the scene reflected in depth about the subject and seen by
// camera B turned the mirrored way gives the same images. Bone lengths cannot tell the two apart (a reflection keeps
// every length) and with narrow or near-parallel cameras reprojection barely can. This builds the twin of a solution
// as LM parameters: the scene is reflected by S about the plane z = depthA in camera A; camera B keeps its image rows
// (R' = S·R·S, t'xy = txy − (R'·c)xy with c = (0, 0, 2·depthA)) and its depth is reflected about the subject's depth
// there, depthB, so the subject stays in front of both cameras.
function neckerTwin({ camA, camB }, depthA, depthB) {
  const S = [[1, 0, 0], [0, 1, 0], [0, 0, -1]];
  const R = matMul(matMul(S, camB.R), S);
  const t = sub3(matVec(S, camB.t), matVec(R, [0, 0, 2 * depthA])).map((v, i) => (i === 2 ? v + 2 * depthB : v));
  return [Math.log(camA.f), Math.log(camB.f), ...vectorFromRotation(R), ...anglesFromUnit(t)];
}

// frames: [{ a, b, shapeA }] — 2D points of both cameras plus camera A's monocular 3D shape (MediaPipe world/relative
// landmarks, any origin and scale, null where absent), which settles the depth reflection.
// bones: [[jointI, jointJ], …] of the tracked skeleton — the moving body is the calibration object.
// groups: optional per-point detection id (e.g. "right"/"left"/"face"); a detection is accepted or rejected as a whole.
export function calibratePair(frames, sizeA, sizeB, options) {
  const {
    bones,
    groups,
    // Webcams sit around 60–80°; the lower end also covers telephoto rigs such as InterHand2.6M (~15°).
    fovRange = [10, 120],
    minVisibility = 0.6,
    thresholdPx = 0.008 * Math.max(sizeA.width, sizeB.width),
    boneSigma = 0.05,
    // 2D keypoints carry several pixels of view-dependent bias; weighting reprojection as 1 px noise lets
    // it overpower bone constancy and pull focals to degenerate extremes (measured on CMU Panoptic pairs).
    reprojSigmaPx = 6,
    maxCorrespondences = 1500,
    maxBoneFrames = 200,
    minBoneFrames = 20,
    minBones = 3,
    maxStarts = 5,
    seed = 1,
  } = options;
  const log = [];
  const corr = collectCorrespondences(frames, minVisibility);
  if (corr.length < 30) throw new Error(`too few correspondences (${corr.length})`);

  // A detection is either the same person/hand at the same moment in both views or garbage as a whole (mirrored
  // pose, swapped hands, tracking glitch, heavy blur, desync). Drop detections where most points miss the epipolar
  // geometry — including their few accidental inliers — then re-fit on what remains.
  const detectionOf = (c) => (groups ? `${c.frame}:${groups[c.joint]}` : String(c.frame));
  const first = ransacFundamental(corr.map((c) => c.a), corr.map((c) => c.b), { thresholdPx, seed });
  const errorsByDetection = new Map();
  for (const c of corr) {
    const key = detectionOf(c);
    if (!errorsByDetection.has(key)) errorsByDetection.set(key, []);
    errorsByDetection.get(key).push(Math.sqrt(sampsonError(first.F, c.a, c.b)));
  }
  const rejected = new Set([...errorsByDetection].filter(([, errs]) => median(errs) > thresholdPx).map(([key]) => key));
  const kept = corr.filter((c) => !rejected.has(detectionOf(c)));
  const { F, inliers: keptInliers } = ransacFundamental(kept.map((c) => c.a), kept.map((c) => c.b), { thresholdPx, seed });
  const inlierCorr = keptInliers.map((i) => kept[i]);
  log.push(`detections rejected: ${rejected.size}/${errorsByDetection.size} | RANSAC: ${inlierCorr.length}/${kept.length} inliers`);

  const rand = mulberry32(seed + 7);
  const pick = (list, n) => (list.length <= n ? list : [...list].sort(() => rand() - 0.5).slice(0, n));
  const inlierKey = new Set(inlierCorr.map((c) => `${c.frame}:${c.joint}`));
  const tracks = bones
    .map((bone) => ({ bone, frames: frames.map((_, k) => k).filter((k) => bone.every((j) => inlierKey.has(`${k}:${j}`))) }))
    .filter((track) => track.frames.length >= minBoneFrames)
    .map((track) => ({ ...track, frames: pick(track.frames, maxBoneFrames) }));
  if (tracks.length < minBones) throw new Error(`only ${tracks.length} bones are seen by both cameras in ≥${minBoneFrames} frames; need ${minBones}`);
  log.push(`bones: ${tracks.map((t) => `${t.bone.join("-")}×${t.frames.length}`).join(" ")}`);

  const cheiralitySet = pick(inlierCorr, 300).map((c) => [c.a, c.b]);
  const candidates = focalValley(F, sizeA, sizeB, { fovRange }).map(({ intrA, intrB, cost }) => {
    const E = matMul(matMul(transpose(intrinsics(intrB.f, intrB.cx, intrB.cy)), F), intrinsics(intrA.f, intrA.cx, intrA.cy));
    const pose = poseFromEssential(E, cheiralitySet, intrA, intrB);
    const camA = { ...intrA, R: identity(3), t: [0, 0, 0] };
    const camB = { ...intrB, R: pose.R, t: pose.t };
    return { intrA, intrB, pose, cost, variation: boneVariation(frames, tracks, camA, camB) };
  });
  // Real 2D keypoints carry view-dependent bias, so the valley often has several comparable dips.
  // Refine from each local minimum of bone variation and let the full objective decide.
  const starts = candidates
    .filter((c, i) => c.variation <= (candidates[i - 1]?.variation ?? Infinity) && c.variation <= (candidates[i + 1]?.variation ?? Infinity))
    .sort((a, b) => a.variation - b.variation)
    .slice(0, maxStarts);
  const fovs = (fa, fb) => `${fovFromFocal(sizeA.width, fa).toFixed(1)}°/${fovFromFocal(sizeB.width, fb).toFixed(1)}°`;
  log.push(`valley starts (FOV_A/FOV_B, bone variation): ${starts.map((c) => `${fovs(c.intrA.f, c.intrB.f)} ${(c.variation * 100).toFixed(1)}%`).join(", ")}`);

  const reprojSet = pick(inlierCorr, maxCorrespondences);

  const reprojCount = reprojSet.length * 4;
  const residuals = (p) => {
    const { camA, camB } = camerasFromParams(p, sizeA, sizeB);
    const out = [];
    for (const c of reprojSet) {
      const X = triangulate([{ p: c.a, cam: camA }, { p: c.b, cam: camB }]);
      const pa = project(X, camA);
      const pb = project(X, camB);
      for (const r of [pa[0] - c.a[0], pa[1] - c.a[1], pb[0] - c.b[0], pb[1] - c.b[1]]) out.push(huber(r / reprojSigmaPx, 2));
    }
    for (const track of tracks) {
      const lengths = trackLengths(frames, track, camA, camB);
      const ref = median(lengths);
      for (const l of lengths) out.push(cauchy((l / ref - 1) / boneSigma, 2));
    }
    return out;
  };

  const split = (p) => {
    const r = residuals(p);
    const sq = (xs) => xs.reduce((acc, v) => acc + v * v, 0).toFixed(0);
    return `reproj ${sq(r.slice(0, reprojCount))} + bones ${sq(r.slice(reprojCount))}`;
  };
  const runs = starts.map(({ intrA, intrB, pose }) =>
    levenbergMarquardt(residuals, [Math.log(intrA.f), Math.log(intrB.f), ...vectorFromRotation(pose.R), ...anglesFromUnit(pose.t)]),
  );
  runs.forEach(({ params, cost }) => log.push(`  LM → ${fovs(Math.exp(params[0]), Math.exp(params[1]))}: ${split(params)} = ${cost.toFixed(0)}`));
  const best = runs.reduce((a, b) => (b.cost < a.cost ? b : a));

  const bestCams = camerasFromParams(best.params, sizeA, sizeB);
  const subject = reprojSet.map((c) => triangulate([{ p: c.a, cam: bestCams.camA }, { p: c.b, cam: bestCams.camB }]));
  const twin = levenbergMarquardt(residuals, neckerTwin(bestCams, median(subject.map((X) => X[2])), median(subject.map((X) => project(X, bestCams.camB)[2]))));
  const pointCount = frames[0].a.length;
  const groupSlots = new Map();
  for (let j = 0; j < pointCount; j++) {
    const name = groups ? groups[j] : "all";
    groupSlots.set(name, [...(groupSlots.get(name) ?? []), j]);
  }
  // share is NaN when nothing reconstructs in front of both cameras — with strong perspective the twin does not exist
  // and LM drives it there.
  const mirrorShare = (params) => {
    const counts = mirrorCounts(frames, reconstruct(frames, camerasFromParams(params, sizeA, sizeB)), groupSlots);
    const sum = Object.values(counts).reduce((s, c) => ({ mirrored: s.mirrored + c.mirrored, total: s.total + c.total }), { mirrored: 0, total: 0 });
    return { share: sum.mirrored / sum.total, text: sum.total ? Object.entries(counts).map(([g, c]) => `${g} ${c.mirrored}/${c.total}`).join(", ") : "nothing in front of both cameras" };
  };
  const own = mirrorShare(best.params);
  if (Number.isNaN(own.share)) throw new Error("no detection has enough 3D shape points to tell the scene from its mirror image");
  const reflected = mirrorShare(twin.params);
  const { params } = reflected.share < own.share ? twin : best;
  log.push(
    `mirror check (MediaPipe 3D shapes fitting better mirrored): solution ${own.text} (cost ${best.cost.toFixed(0)}), depth-reflected twin ${reflected.text} (cost ${twin.cost.toFixed(0)}) → ${params === best.params ? "solution" : "twin"}`,
  );
  const { camA, camB } = camerasFromParams(params, sizeA, sizeB);

  const reprojErrors = reprojSet.map((c) => {
    const X = triangulate([{ p: c.a, cam: camA }, { p: c.b, cam: camB }]);
    return (Math.hypot(...sub3([...project(X, camA).slice(0, 2), 0], [c.a[0], c.a[1], 0])) + Math.hypot(...sub3([...project(X, camB).slice(0, 2), 0], [c.b[0], c.b[1], 0]))) / 2;
  });
  const sorted = [...reprojErrors].sort((x, y) => x - y);
  log.push(`refined: FOV_A ${fovFromFocal(sizeA.width, camA.f).toFixed(1)}°, FOV_B ${fovFromFocal(sizeB.width, camB.f).toFixed(1)}°, median reprojection ${sorted[Math.floor(sorted.length / 2)].toFixed(2)} px`);

  return { camA, camB, log, rejectedDetections: [...rejected], inlierCount: inlierCorr.length, correspondenceCount: corr.length, medianReprojPx: sorted[Math.floor(sorted.length / 2)] };
}

// Triangulate every joint visible in both views; null where it is not or where the two views disagree.
export function reconstruct(frames, { camA, camB }, { minVisibility = 0.5, maxReprojPx = 15 } = {}) {
  return frames.map((frame) =>
    frame.a.map((pa, j) => {
      const pb = frame.b[j];
      if (pa[2] < minVisibility || pb[2] < minVisibility) return null;
      const X = triangulate([{ p: pa, cam: camA }, { p: pb, cam: camB }]);
      const ra = project(X, camA);
      const rb = project(X, camB);
      if (ra[2] <= 0 || rb[2] <= 0) return null;
      if (Math.max(Math.hypot(ra[0] - pa[0], ra[1] - pa[1]), Math.hypot(rb[0] - pb[0], rb[1] - pb[1])) > maxReprojPx) return null;
      return X;
    }),
  );
}

// Similarity (Umeyama) alignment dst ≈ s·R·src + t over paired 3D points.
export function umeyama(src, dst, withScale = true) {
  const n = src.length;
  const mean = (pts) => [0, 1, 2].map((i) => pts.reduce((s, p) => s + p[i], 0) / n);
  const ms = mean(src);
  const md = mean(dst);
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let varS = 0;
  for (let k = 0; k < n; k++) {
    const a = sub3(src[k], ms);
    const b = sub3(dst[k], md);
    varS += dot3(a, a);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += b[i] * a[j];
  }
  const { U, S, V } = svd3(cov.map((row) => row.map((v) => v / n)));
  const d = det3(U) * det3(V) < 0 ? -1 : 1;
  const D = [[1, 0, 0], [0, 1, 0], [0, 0, d]];
  const R = matMul(matMul(U, D), transpose(V));
  const s = withScale ? ((S[0] + S[1] + d * S[2]) * n) / varS : 1;
  const t = sub3(md, matVec(R, ms).map((v) => v * s));
  return { s, R, t, apply: (p) => matVec(R, p).map((v, i) => s * v + t[i]) };
}
