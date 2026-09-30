# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""Cut a synced window from CMU Panoptic 171204_pose1, undistort to pinhole, downscale, dump GT for eval.

GT (intrinsics, extrinsics, 3D joints) is written only for evaluation; the stereo pipeline never reads it.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SEQ = "171204_pose1"
URL = f"http://domedb.perception.cs.cmu.edu/webdata/dataset/{SEQ}/videos/hd_shared_crf20/hd_{{cam}}.mp4"
CAMS = ["00_00", "00_29", "00_21", "00_11", "00_12"]
START, COUNT, FPS = 3000, 600, 30000 / 1001
SRC_W, SRC_H, SCALE = 1920, 1080, 0.5
ROOT = Path(__file__).parent / "panoptic-pose1"
OUT = ROOT / "clip"


def load_cameras() -> dict[str, dict]:
    cal = json.loads((ROOT / "calibration.json").read_text())
    return {c["name"]: c for c in cal["cameras"] if c["type"] == "hd" and c["name"] in CAMS}


def load_gt() -> list[list[list[float]] | None]:
    frames = []
    for i in range(START, START + COUNT):
        bodies = json.loads((ROOT / "hdPose3d_stage1_coco19" / f"body3DScene_{i:08d}.json").read_text())["bodies"]
        frames.append(np.array(bodies[0]["joints19"]).reshape(19, 4).round(3).tolist() if len(bodies) == 1 else None)
    return frames


def cut_camera(name: str, cam: dict) -> None:
    K = np.array(cam["K"], dtype=np.float64)
    dist = np.array(cam["distCoef"], dtype=np.float64)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (SRC_W, SRC_H), cv2.CV_32FC1)
    w, h = int(SRC_W * SCALE), int(SRC_H * SCALE)
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{START / FPS:.6f}", "-i", URL.format(cam=name),
         "-frames:v", str(COUNT), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "30000/1001",
         "-i", "-", "-c:v", "libx264", "-crf", "20", "-g", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(OUT / f"cam_{name}.mp4")],
        stdin=subprocess.PIPE,
    )
    assert reader.stdout and writer.stdin
    frame_bytes = SRC_W * SRC_H * 3
    n = 0
    while buf := reader.stdout.read(frame_bytes):
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(SRC_H, SRC_W, 3)
        und = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        small = cv2.resize(und, (w, h), interpolation=cv2.INTER_AREA)
        if n == 0:
            cv2.imwrite(str(OUT / f"first_{name}.png"), small)
        writer.stdin.write(small.tobytes())
        n += 1
    writer.stdin.close()
    if reader.wait() != 0 or writer.wait() != 0 or n != COUNT:
        sys.exit(f"{name}: got {n}/{COUNT} frames, reader={reader.returncode} writer={writer.returncode}")
    print(f"{name}: {n} frames", flush=True)


def scaled_camera(cam: dict) -> dict:
    K = np.array(cam["K"]) * SCALE
    K[2, 2] = 1
    return {
        "width": int(SRC_W * SCALE), "height": int(SRC_H * SCALE),
        "K": K.round(4).tolist(), "R": cam["R"], "t": [v[0] for v in cam["t"]],
    }


def draw_gt_overlay(name: str, cam: dict, joints: list[list[float]]) -> None:
    img = cv2.imread(str(OUT / f"first_{name}.png"))
    K, R, t = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"])
    X = np.array(joints)[:, :3]
    x = (K @ (R @ X.T + t[:, None])).T
    for u, v in x[:, :2] / x[:, 2:]:
        cv2.circle(img, (int(u), int(v)), 4, (0, 0, 255), -1)
    cv2.imwrite(str(OUT / f"check_{name}.png"), img)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cams = load_cameras()
    gt = load_gt()
    scaled = {n: scaled_camera(cams[n]) for n in CAMS}
    (OUT / "gt.json").write_text(json.dumps({
        "sequence": SEQ, "startFrame": START, "fps": FPS, "units": "cm", "jointSet": "coco19",
        "cameras": scaled, "joints": gt,
    }))
    print(f"gt: {sum(f is not None for f in gt)}/{COUNT} single-body frames", flush=True)
    for name in sys.argv[1:] or CAMS:
        cut_camera(name, cams[name])
        if gt[0] is not None:
            draw_gt_overlay(name, scaled[name], gt[0])


if __name__ == "__main__":
    main()
