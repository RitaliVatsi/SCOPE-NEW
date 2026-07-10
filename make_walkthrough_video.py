"""
Stitch the per-step egocentric frames saved by run_goatbench_evaluation.py /
run_object_goal.py (in an episode's "snapshot/" folder, named "{step}-view_{idx}.png")
into a first-person walkthrough mp4.

Usage:
    python make_walkthrough_video.py --episode_dir theirresult/r2-object/00824-Dd4bFSTQ8gi_ep_0_objectgoal --out theirresult/r2-object/first_person_walkthrough.mp4
"""

import argparse
import glob
import os
import re

import cv2

FRAME_RE = re.compile(r"^(\d+)-view_(\d+)\.png$")


def sorted_frames(snapshot_dir):
    frames = []
    for path in glob.glob(os.path.join(snapshot_dir, "*-view_*.png")):
        m = FRAME_RE.match(os.path.basename(path))
        if not m:
            continue
        step, view_idx = int(m.group(1)), int(m.group(2))
        frames.append((step, view_idx, path))
    frames.sort(key=lambda x: (x[0], x[1]))
    return [path for _, _, path in frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", required=True, help="episode output directory containing a snapshot/ subfolder")
    parser.add_argument("--out", required=True, help="output mp4 path")
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    snapshot_dir = os.path.join(args.episode_dir, "snapshot")
    frame_paths = sorted_frames(snapshot_dir)
    if not frame_paths:
        raise SystemExit(f"No '*-view_*.png' frames found in {snapshot_dir}")

    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    for path in frame_paths:
        frame = cv2.imread(path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()

    print(f"Wrote {len(frame_paths)} frames -> {args.out} ({width}x{height} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
