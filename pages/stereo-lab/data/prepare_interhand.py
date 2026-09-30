# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""Turn the first InterHand2.6M 5 fps sequence (streamed prefix of the image tar) into the lab's clip format.

Each camera image is shifted so its principal point lands on the image centre — the lab's camera model assumes a
centred principal point, as webcams roughly have. GT (intrinsics, extrinsics, 3D joints) is written for evaluation only.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent / "interhand"
CAPTURE = "1"
SEQ = "ROM07_Rt_Finger_Occlusions"
IMAGES = ROOT / "InterHand2.6M_5fps_batch1" / "images" / "test" / f"Capture{CAPTURE}" / SEQ
OUT = ROOT / "clip"
FPS = 5


def main() -> None:
    cams_meta = json.loads((ROOT / "annotations" / "test_camera.json").read_text())[CAPTURE]
    joints = json.loads((ROOT / "annotations" / "test_joint_3d.json").read_text())[CAPTURE]
    cam_dirs = sorted(p for p in IMAGES.iterdir() if p.is_dir())
    frame_sets = [{int(f.stem.removeprefix("image")) for f in d.glob("image*.jpg")} for d in cam_dirs]
    frames = sorted(set.intersection(*[s for s in frame_sets if len(s) == max(map(len, frame_sets))]))
    cams = [d.name.removeprefix("cam") for d, s in zip(cam_dirs, frame_sets) if set(frames) <= s]
    print(f"{len(cams)} cameras with all {len(frames)} frames; skipped {[d.name for d, s in zip(cam_dirs, frame_sets) if not set(frames) <= s]}")
    OUT.mkdir(exist_ok=True)

    cameras = {}
    for cam in cams:
        first = cv2.imread(str(IMAGES / f"cam{cam}" / f"image{frames[0]}.jpg"))
        h, w = first.shape[:2]
        (fx, fy), (cx, cy) = cams_meta["focal"][cam], cams_meta["princpt"][cam]
        shift = np.float32([[1, 0, w / 2 - cx], [0, 1, h / 2 - cy]])
        writer = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-crf", "16", "-g", "5", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / f"cam_{cam}.mp4")],
            stdin=subprocess.PIPE,
        )
        assert writer.stdin
        for i, frame in enumerate(frames):
            img = cv2.warpAffine(cv2.imread(str(IMAGES / f"cam{cam}" / f"image{frame}.jpg")), shift, (w, h))
            if i == 0:
                cv2.imwrite(str(OUT / f"first_{cam}.png"), img)
            writer.stdin.write(img.tobytes())
        writer.stdin.close()
        if writer.wait() != 0:
            sys.exit(f"ffmpeg failed for {cam}")
        R = np.array(cams_meta["camrot"][cam])
        t = -R @ np.array(cams_meta["campos"][cam])
        cameras[cam] = {"width": w, "height": h, "K": [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], "R": R.tolist(), "t": t.tolist()}

    gt = []
    for frame in frames:
        f = joints[str(frame)]
        valid = np.array(f["joint_valid"]).reshape(-1)
        gt.append([[*xyz, float(v)] for xyz, v in zip(f["world_coord"], valid)])
    (OUT / "gt.json").write_text(json.dumps({
        "dataset": "InterHand2.6M", "sequence": f"Capture{CAPTURE}/{SEQ}", "frames": frames, "fps": FPS, "units": "mm",
        "jointSet": "interhand42", "cameras": cameras, "joints": gt,
    }))

    for cam in cams[:4]:
        c = cameras[cam]
        img = cv2.imread(str(OUT / f"first_{cam}.png"))
        X = np.array(gt[0])[:, :3]
        x = (np.array(c["K"]) @ (np.array(c["R"]) @ X.T + np.array(c["t"])[:, None])).T
        for (u, v), ok in zip(x[:, :2] / x[:, 2:], np.array(gt[0])[:, 3]):
            if ok:
                cv2.circle(img, (int(u), int(v)), 3, (0, 0, 255), -1)
        cv2.imwrite(str(OUT / f"check_{cam}.png"), img)
    print("done:", OUT)


if __name__ == "__main__":
    main()
