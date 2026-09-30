// Runs calibratePair off the main thread: it takes seconds, and live mode calibrates in the background while tracking.
import { calibratePair } from "./stereo.js";

onmessage = ({ data: { id, frames, sizeA, sizeB, options } }) => {
  try {
    postMessage({ id, calib: calibratePair(frames, sizeA, sizeB, options) });
  } catch (e) {
    postMessage({ id, error: e.message });
  }
};
