"""Manual teleoperation with on-the-go mental-map building and landmark
validation. The agent starts with no prior map of the scene: you drive it
around, mark landmarks as you pass them, and the system builds a topological
map from the actual trajectory and confirms arrival at each landmark in the
order it was marked.

Run (from the SCOPE/ directory, inside the `scope` conda env):
    python -m freshwork.teleop_run

Controls:
    w / x     move forward / backward
    a / d     turn left / right
    s         mark the current position as the 'start' landmark
    e         mark the current position as the 'end' landmark
    l         mark a landmark at the current position (label is auto-
              suggested from the semantic sensor at the view center;
              press Enter to accept it or type your own)
    v         manually re-check distance to the active landmark
    q / ESC   quit, saving the mental map (json) and a top-down plot (png)
"""
import os
import time

import cv2
import numpy as np

from freshwork.mental_map import MentalMap
from freshwork.sim_env import TeleopSimEnv

FRESHWORK_DIR = os.path.dirname(os.path.abspath(__file__))
SCOPE_DIR = os.path.dirname(FRESHWORK_DIR)

SCENE_ID = "00824-Dd4bFSTQ8gi"
SCENE_DATA_PATH = "/media/krishna/5a81cf60-33b9-4686-8d6b-339147b7004c1/SCOPE/data/hm3d_raw/versioned_data/hm3d-0.2/hm3d"
SCENE_DATASET_CONFIG = os.path.join(SCOPE_DIR, "data", "hm3d_annotated_basis.scene_dataset_config.json")
ARRIVAL_RADIUS = 1.0
RESULTS_DIR = os.path.join(FRESHWORK_DIR, "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = TeleopSimEnv(
        scene_id=SCENE_ID,
        scene_data_path=SCENE_DATA_PATH,
        scene_dataset_config_path=SCENE_DATASET_CONFIG,
    )
    mmap = MentalMap(arrival_radius=ARRIVAL_RADIUS)

    pos_normal, yaw, _ = env.pose()
    mmap.observe_pose(pos_normal, yaw)

    print(__doc__)
    obs = env.observations()

    try:
        while True:
            canvas = np.ascontiguousarray(obs["color_sensor"][:, :, :3][:, :, ::-1])
            active = mmap.active_landmark
            status = f"active landmark: {active.label}" if active else "active landmark: none (all reached)"
            cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("freshwork teleop", canvas)
            key = cv2.waitKey(0) & 0xFF

            action = None
            if key == ord("w"):
                action = "move_forward"
            elif key == ord("x"):
                action = "move_backward"
            elif key == ord("a"):
                action = "turn_left"
            elif key == ord("d"):
                action = "turn_right"
            elif key == ord("s"):
                pos_normal, yaw, _ = env.pose()
                mmap.add_landmark("start", pos_normal)
                print(f"marked 'start' at {pos_normal}")
                continue
            elif key == ord("e"):
                pos_normal, yaw, _ = env.pose()
                mmap.add_landmark("end", pos_normal)
                print(f"marked 'end' at {pos_normal}")
                continue
            elif key == ord("l"):
                suggested = env.label_at_center(obs) or "landmark"
                label = input(f"landmark label [{suggested}]: ").strip() or suggested
                pos_normal, yaw, _ = env.pose()
                mmap.add_landmark(label, pos_normal)
                print(f"marked landmark '{label}' at {pos_normal}")
                continue
            elif key == ord("v"):
                pos_normal, yaw, _ = env.pose()
                reached = mmap.check_arrival(pos_normal)
                if reached:
                    print(f"reached landmark '{reached.label}'")
                else:
                    dist = mmap.distance_to_active(pos_normal)
                    if dist is not None:
                        print(f"not yet at '{mmap.active_landmark.label}', distance={dist:.2f}m")
                    else:
                        print("no active landmark")
                continue
            elif key in (ord("q"), 27):
                break
            else:
                continue

            obs = env.step(action)
            pos_normal, yaw, _ = env.pose()
            mmap.observe_pose(pos_normal, yaw)
            reached = mmap.check_arrival(pos_normal)
            if reached:
                print(f"reached landmark '{reached.label}'")
    finally:
        cv2.destroyAllWindows()
        stamp = int(time.time())
        map_path = os.path.join(RESULTS_DIR, f"mental_map_{stamp}.json")
        mmap.save(map_path)
        print(f"saved mental map to {map_path}")
        try:
            plot_path = map_path.replace(".json", ".png")
            _plot_topdown(mmap, plot_path)
            print(f"saved trajectory plot to {plot_path}")
        except Exception as e:
            print(f"skipped plot: {e}")
        env.close()


def _plot_topdown(mmap: MentalMap, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [n.position[0] for n in mmap.nodes]
    ys = [n.position[1] for n in mmap.nodes]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, "-o", markersize=2, color="steelblue", label="trajectory")
    for lm in mmap.landmarks:
        color = "green" if lm.reached else "red"
        plt.scatter([lm.position[0]], [lm.position[1]], color=color, s=80, marker="*")
        plt.annotate(lm.label, (lm.position[0], lm.position[1]))
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("Mental map (top-down)")
    plt.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
