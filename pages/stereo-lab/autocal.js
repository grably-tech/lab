// Auto-calibration policy for live mode (pure): when the collected frames are varied enough to calibrate from, and when
// a calibration has stopped matching the cameras (one was moved). The shell collects frames and runs calibratePair.
import { median, project, triangulate } from "./stereo.js";

export const MIN_FRAMES = 150;
// Calibration points must span this share of each image's width and height: a still face is a small, nearly planar
// patch that constrains the geometry poorly (~0.08–0.10), while a face+hands session that calibrated correctly spanned
// 0.20.
export const MIN_SPREAD = 0.15;
// Moved-camera checks over the last DRIFT_FRAMES frames. Reprojection catches tilt and roll of a camera (on a real
// session: 0.5° tilt → 3 px, 1° roll → 4 px against 0.8 px normally), but not a turn about the axis perpendicular to
// the baseline: that only slides points along the epipolar lines. Such a turn shrinks or grows the reconstruction
// instead (≈2% face width per degree, ±2% normally), so the face width is checked against the calibration's.
export const DRIFT_FRAMES = 30;
export const DRIFT_FACTOR = 3;
export const DRIFT_MIN_PX = 3;
export const DRIFT_SCALE = 0.08;
const MIN_VISIBILITY = 0.6;

// Spread of the points seen in one camera: 5th–95th percentile extent over width and height, as a share of the image;
// the smaller of the two.
function spread(points, size) {
  if (points.length < 2) return 0;
  const extent = (values, full) => {
    const v = [...values].sort((a, b) => a - b);
    return (v[Math.floor(v.length * 0.95)] - v[Math.floor(v.length * 0.05)]) / full;
  };
  return Math.min(extent(points.map((p) => p[0]), size.width), extent(points.map((p) => p[1]), size.height));
}

// frames: [{ a, b }] of calibration points. progress is 0…1 towards ready.
export function readiness(frames, sizeA, sizeB) {
  const both = (side) => frames.flatMap((f) => f[side].filter((p, j) => p[2] >= MIN_VISIBILITY && f[side === "a" ? "b" : "a"][j][2] >= MIN_VISIBILITY));
  const minSpread = Math.min(spread(both("a"), sizeA), spread(both("b"), sizeB));
  return {
    frames: frames.length,
    spread: minSpread,
    ready: frames.length >= MIN_FRAMES && minSpread >= MIN_SPREAD,
    progress: Math.min(1, frames.length / MIN_FRAMES, minSpread / MIN_SPREAD),
  };
}

// Median reprojection error (px, mean of both views) of one frame's points seen by both cameras; null if none.
export function frameReprojectionPx(frame, { camA, camB }) {
  const errors = [];
  frame.a.forEach((pa, j) => {
    const pb = frame.b[j];
    if (pa[2] < MIN_VISIBILITY || pb[2] < MIN_VISIBILITY) return;
    const X = triangulate([{ p: pa, cam: camA }, { p: pb, cam: camB }]);
    const ra = project(X, camA);
    const rb = project(X, camB);
    errors.push((Math.hypot(ra[0] - pa[0], ra[1] - pa[1]) + Math.hypot(rb[0] - pb[0], rb[1] - pb[1])) / 2);
  });
  return errors.length ? median(errors) : null;
}

// errors: recent per-frame reprojection errors under the current calibration; the calibration no longer matches the
// cameras when their median exceeds DRIFT_FACTOR× the error it had when it was made. widths: recent stereo face widths;
// it no longer matches when their median is off the calibration's face width by more than DRIFT_SCALE (skipped when the
// calibration saw no face).
export function drift({ errors, widths }, { medianReprojPx, faceWidthUnits }) {
  const recentErrors = errors.slice(-DRIFT_FRAMES);
  const recentWidths = widths.slice(-DRIFT_FRAMES);
  const errorPx = recentErrors.length ? median(recentErrors) : null;
  const limitPx = Math.max(DRIFT_FACTOR * medianReprojPx, DRIFT_MIN_PX);
  const scale = faceWidthUnits && recentWidths.length ? median(recentWidths) / faceWidthUnits : null;
  const reprojection = recentErrors.length >= DRIFT_FRAMES && errorPx > limitPx;
  const size = scale !== null && recentWidths.length >= DRIFT_FRAMES && Math.abs(scale - 1) > DRIFT_SCALE;
  return { errorPx, limitPx, scale, reprojection, size, drifted: reprojection || size };
}
