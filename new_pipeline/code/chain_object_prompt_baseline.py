"""
Chain-of-landmarks object-goal exploration ("mental map" mode).

Instead of a single target word (run_object_goal.py) or a goal image, this script takes
a free-text navigation instruction that mentions one or more intermediate landmarks
before the final target, e.g.:

    "Head toward the couch in the living room, then pass by the rug, and finally find
    the dining table."

The instruction is parsed (one LLM call) into an ordered chain of landmark names plus a
final target name. The agent then explores landmark-by-landmark: at each point in the
chain, the VLM is only asked to find the *current* landmark, biasing frontier/snapshot
selection toward it. A landmark is only marked "found" once the VLM has committed to a
Snapshot for it, self-refine has confirmed that choice, and the agent has actually
navigated to and arrived at it - the same rigorous path used for the final target, applied
uniformly to every landmark in the chain. Only reaching the final entry (the actual
target) ends the run successfully.

There is no GOAT-Bench ground truth for a free-typed instruction like this, so - same as
run_object_goal.py - this does not report success rate / SPL, only the trajectory and a
mental_map_result.json recording which landmarks were found, in what order, at which step.
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import math
import time
import json
import re
import logging
import matplotlib.pyplot as plt
import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene_goatbench import Scene
from src.utils import resize_image, get_pts_angle_goatbench
from src.query_vlm_goatbench import query_vlm_for_response
from src.eval_utils_gpt_goatbench import call_openai_api
from src.logger_goatbench import Logger
from src.potential_graph import PotentialGraph
from src.potential_estimation_gpt_goal import get_potential_estimation


def parse_landmark_chain(chain_prompt):
    """Ask the VLM to split a free-text route instruction into an ordered list of
    intermediate landmarks plus the final target object. Returns (landmarks, target)."""
    sys_prompt = (
        "You extract navigation landmarks from an indoor route instruction. "
        "Read the instruction and list, in the order they are meant to be visited, "
        "every intermediate landmark object mentioned, followed by the final target "
        "object (the thing the instruction ultimately wants found). "
        "Respond with strict JSON only, no other text, in this exact form: "
        '{"landmarks": ["<landmark1>", "<landmark2>", ...], "target": "<final target>"}. '
        "Each entry should be a short, simple object noun (e.g. 'couch', 'rug', 'dining table'), "
        "not a full sentence. If no intermediate landmarks are mentioned, return an empty list "
        "for landmarks and just the target."
    )
    response = call_openai_api(sys_prompt, [(f"Instruction: {chain_prompt}",)])
    if response is None:
        raise RuntimeError("Failed to parse landmark chain: no response from VLM")

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find JSON in VLM response: {response}")
    parsed = json.loads(match.group(0))
    landmarks = [str(x).strip().lower() for x in parsed.get("landmarks", [])]
    target = str(parsed["target"]).strip().lower()
    return landmarks, target


def build_mental_map(landmarks, target):
    mental_map = [{"name": name, "is_target": False, "found": False, "found_step": None} for name in landmarks]
    mental_map.append({"name": target, "is_target": True, "found": False, "found_step": None})
    return mental_map


def current_active_entry(mental_map):
    for entry in mental_map:
        if not entry["found"]:
            return entry
    return mental_map[-1]


def main(cfg, chain_prompt, split=1):
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logging.info(f"Chain prompt: {chain_prompt}")
    landmarks, target = parse_landmark_chain(chain_prompt)
    mental_map = build_mental_map(landmarks, target)
    logging.info(f"Parsed landmark chain: {landmarks} -> target: {target}")

    scene_data_list = sorted(os.listdir(cfg.test_data_dir))
    scene_data_file = scene_data_list[0]
    scene_name = scene_data_file.split(".")[0]

    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
        cfg.scene_data_path + "/val"
    )
    scene_id = [s for s in all_scene_ids if scene_name in s][0]

    scene_data = json.load(open(os.path.join(cfg.test_data_dir, scene_data_file), "r"))
    episode = scene_data["episodes"][split - 1]
    episode_id = episode["episode_id"]

    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    logger = Logger(cfg.output_dir, 0.0, 1.0, split, voxel_size=cfg.tsdf_grid_size)

    logging.info(f"Loading scene {scene_id}")
    scene = Scene(
        scene_id,
        cfg,
        cfg_cg,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
    )

    pts, angle = get_pts_angle_goatbench(
        episode["start_position"], episode["start_rotation"]
    )

    floor_height = pts[1]
    tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
    num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
    num_step = max(num_step, 50)
    tsdf_planner = TSDFPlanner(
        vol_bnds=tsdf_bnds,
        voxel_size=cfg.tsdf_grid_size,
        floor_height=floor_height,
        floor_height_offset=0,
        pts_init=pts,
        init_clearance=cfg.init_clearance * 2,
        save_visualization=cfg.save_visualization,
    )

    potential_graph = PotentialGraph(
        vol_bounds=tsdf_bnds,
        voxel_size=cfg.tsdf_grid_size,
        grid_resolution=getattr(cfg, "potential_grid_resolution", 1.0),
        decay_factor=getattr(cfg, "potential_decay_factor", 0.95),
        influence_radius=getattr(cfg, "potential_influence_radius", 3.0),
    )

    target_slug = target.replace(" ", "_")
    subtask_id = f"{scene_id}_{episode_id}_chainprompt_{target_slug}"
    episode_dir, eps_frontier_dir, eps_snapshot_dir, eps_potential_dir = logger.init_episode(
        episode_id=f"{scene_id}_ep_{episode_id}_chainprompt"
    )
    eps_potential_dir = os.path.join(episode_dir, "potential_graph")
    os.makedirs(eps_potential_dir, exist_ok=True)

    logging.info(f"\n\nScene {scene_id} initialization successful!")

    logger.subtask_object_observe_dir = os.path.join(
        logger.output_dir, subtask_id, "object_observations"
    )
    if os.path.exists(logger.subtask_object_observe_dir):
        os.system(f"rm -r {logger.subtask_object_observe_dir}")
    os.makedirs(logger.subtask_object_observe_dir, exist_ok=False)
    logger.pts_voxels = np.vstack(
        [np.empty((0, 2)), tsdf_planner.habitat2voxel(pts)[:2]]
    )
    logger.subtask_explore_dist = 0.0

    goal_obj_ids_mapping = {}

    task_success = False
    cnt_step = -1
    n_filtered_snapshots = 0
    global_step = -1
    max_point_choice = None

    while cnt_step < num_step - 1:
        cnt_step += 1
        global_step += 1

        active_entry = current_active_entry(mental_map)
        subtask_metadata = {
            "question_id": subtask_id,
            "question": f"Can you find the {active_entry['name']}?",
            "image": None,
            "answer": active_entry["name"],
            "goal_obj_ids": [],
            "class": active_entry["name"],
            "goal_positions_voxel": [],
            "task_type": "object",
            "viewpoints": [],
            "gt_subtask_explore_dist": None,
        }

        logging.info(f"\n== step: {cnt_step}, global step: {global_step} ==")
        logging.info(f"Current mental-map target: '{active_entry['name']}' (is_target={active_entry['is_target']})")

        if cnt_step == 0:
            angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
            total_views = 1 + cfg.extra_view_phase_2
        else:
            angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
            total_views = 1 + cfg.extra_view_phase_1
        all_angles = [
            angle + angle_increment * (i - total_views // 2)
            for i in range(total_views)
        ]
        main_angle = all_angles.pop(total_views // 2)
        all_angles.append(main_angle)

        rgb_egocentric_views = []
        all_added_obj_ids = []
        for view_idx, ang in enumerate(all_angles):
            obs, cam_pose = scene.get_observation(pts, angle=ang)
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            semantic_obs = obs["semantic_sensor"]

            obs_file_name = f"{global_step}-view_{view_idx}.png"
            with torch.no_grad():
                annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                    scene.update_scene_graph(
                        image_rgb=rgb[..., :3],
                        depth=depth,
                        intrinsics=cam_intr,
                        cam_pos=cam_pose,
                        pts=pts,
                        pts_voxel=tsdf_planner.habitat2voxel(pts),
                        img_path=obs_file_name,
                        frame_idx=cnt_step * total_views + view_idx,
                        semantic_obs=semantic_obs,
                        gt_target_obj_ids=subtask_metadata["goal_obj_ids"],
                    )
                )
                scene.all_observations[obs_file_name] = rgb
                rgb_egocentric_views.append(
                    resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                )
                if cfg.save_visualization:
                    plt.imsave(
                        os.path.join(eps_snapshot_dir, obs_file_name),
                        annotated_rgb,
                    )
                else:
                    plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)
                for gt_goal_id, det_goal_id in target_obj_id_mapping.items():
                    goal_obj_ids_mapping.setdefault(gt_goal_id, []).append(det_goal_id)
                all_added_obj_ids += added_obj_ids

            scene.periodic_cleanup_objects(
                frame_idx=cnt_step * total_views + view_idx,
                pts=pts,
                goal_obj_ids_mapping=goal_obj_ids_mapping,
            )

            tsdf_planner.integrate(
                color_im=rgb,
                depth_im=depth,
                cam_intr=cam_intr,
                cam_pose=pose_habitat_to_tsdf(cam_pose),
                obs_weight=1.0,
                margin_h=int(cfg.margin_h_ratio * img_height),
                margin_w=int(cfg.margin_w_ratio * img_width),
                explored_depth=cfg.explored_depth,
            )

        all_added_obj_ids = [
            obj_id for obj_id in all_added_obj_ids if obj_id in scene.objects
        ]
        for obj_id, obj in scene.objects.items():
            if (
                np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                < cfg.scene_graph.obj_include_dist + 0.5
            ):
                all_added_obj_ids.append(obj_id)
        scene.update_snapshots(
            obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
        )
        logging.info(
            f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
        )

        update_success = tsdf_planner.update_frontier_map(
            pts=pts,
            cfg=cfg.planner,
            scene=scene,
            cnt_step=cnt_step,
            save_frontier_image=cfg.save_visualization,
            eps_frontier_dir=eps_frontier_dir,
            prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
        )
        if not update_success:
            logging.info("Warning! Update frontier map failed!")

        if len(tsdf_planner.frontiers) > 0:
            if not hasattr(potential_graph, "_analyzed_frontiers"):
                potential_graph._analyzed_frontiers = set()

            for i, frontier in enumerate(tsdf_planner.frontiers):
                frontier_key = (tuple(frontier.position), frontier.image)
                if frontier_key in potential_graph._analyzed_frontiers:
                    continue

                if frontier.feature is not None and getattr(
                    cfg, "enable_potential_estimation", True
                ):
                    try:
                        potential_text = get_potential_estimation(
                            subtask_metadata, frontier.feature
                        )
                        potential_graph.update_from_frontier(
                            frontier=frontier,
                            subtask_metadata=subtask_metadata,
                            occupied_map=None,
                            potential_text=potential_text,
                        )
                        potential_graph._analyzed_frontiers.add(frontier_key)
                    except Exception as e:
                        logging.warning(f"Failed to get potential estimation for frontier {i}: {e}")
                        potential_graph.update_from_frontier(
                            frontier=frontier,
                            subtask_metadata=subtask_metadata,
                            occupied_map=None,
                            potential_text=None,
                        )

        if cfg.choose_every_step:
            if (
                tsdf_planner.max_point is not None
                and type(tsdf_planner.max_point) == Frontier
            ):
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None

        if (
            tsdf_planner.max_point is None
            and tsdf_planner.target_point is None
        ):
            if len(scene.snapshots) == 0 and len(tsdf_planner.frontiers) == 0:
                logging.warning(f"No snapshots or frontiers available for VLM query at step {cnt_step}")
                continue

            logging.info(f"Querying VLM with {len(scene.snapshots)} snapshots and {len(tsdf_planner.frontiers)} frontiers for landmark '{active_entry['name']}'")

            try:
                vlm_response = query_vlm_for_response(
                    subtask_metadata=subtask_metadata,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    rgb_egocentric_views=rgb_egocentric_views,
                    cfg=cfg,
                    verbose=True,
                    potential_graph=potential_graph,
                )
            except Exception as e:
                logging.error(f"Exception during VLM query: {e}")
                vlm_response = None

            if vlm_response is None:
                logging.error(f"Subtask id {subtask_id} invalid: query_vlm_for_response failed!")
                break

            max_point_choice, n_filtered_snapshots = vlm_response

            update_success = tsdf_planner.set_next_navigation_point(
                choice=max_point_choice,
                pts=pts,
                objects=scene.objects,
                cfg=cfg.planner,
                pathfinder=scene.pathfinder,
            )
            if not update_success:
                logging.info(f"Subtask id {subtask_id} invalid: set_next_navigation_point failed!")
                break

        return_values = tsdf_planner.agent_step(
            pts=pts,
            angle=angle,
            objects=scene.objects,
            snapshots=scene.snapshots,
            pathfinder=scene.pathfinder,
            cfg=cfg.planner,
            path_points=None,
            save_visualization=cfg.save_visualization,
        )
        if return_values[0] is None:
            logging.info(f"Subtask id {subtask_id} invalid: agent_step failed!")
            break

        pts, angle, pts_voxel, fig, _, target_arrived = return_values
        logger.log_step(pts_voxel=pts_voxel)
        logging.info(f"Current position: {pts}, {logger.subtask_explore_dist:.3f}")

        potential_graph.mark_visited(pts, radius=1.5)
        scene.sanity_check(cfg=cfg)

        if cfg.save_visualization:
            logger.save_topdown_visualization(
                global_step=global_step,
                subtask_id=subtask_id,
                subtask_metadata=subtask_metadata,
                goal_obj_ids_mapping=goal_obj_ids_mapping,
                fig=fig,
            )
            logger.save_frontier_visualization(
                global_step=global_step,
                subtask_id=subtask_id,
                tsdf_planner=tsdf_planner,
                max_point_choice=max_point_choice,
                global_caption=f"{subtask_metadata['question']}\nlandmark {mental_map.index(active_entry) + 1}/{len(mental_map)}",
            )

            potential_viz_path = os.path.join(
                eps_potential_dir, f"potential_{global_step}_{subtask_id}.png"
            )
            potential_graph.visualize(
                save_path=potential_viz_path,
                title=f"Step {cnt_step} - {subtask_metadata['question'][:50]}...",
            )

        # arriving at the chosen snapshot marks the *current* landmark found, not
        # necessarily the whole chain - only ending the run if it was the final target
        if type(max_point_choice) == SnapShot and target_arrived:
            obs, _ = scene.get_observation(pts, angle=angle)
            rgb = obs["color_sensor"]
            landmark_slug = active_entry["name"].replace(" ", "_")
            plt.imsave(
                os.path.join(logger.subtask_object_observe_dir, f"{landmark_slug}.png"),
                rgb,
            )
            snapshot_filename = max_point_choice.image.split(".")[0]
            os.system(
                f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(logger.subtask_object_observe_dir, f'snapshot_{landmark_slug}_{snapshot_filename}.png')}"
            )

            if not active_entry["found"]:
                active_entry["found"] = True
                active_entry["found_step"] = global_step
            logging.info(f"Mental map: arrived at landmark '{active_entry['name']}' at step {global_step}")

            if active_entry["is_target"]:
                task_success = True
                break

            # move to the next landmark in the chain
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None

    if cfg.save_visualization:
        final_potential_path = os.path.join(eps_potential_dir, f"final_potential_{subtask_id}.png")
        potential_graph.visualize(
            save_path=final_potential_path,
            title=f"Final - chain prompt for '{target}'",
        )
        graph_state_path = os.path.join(eps_potential_dir, f"potential_state_{subtask_id}.pkl")
        potential_graph.save_state(graph_state_path)

    result_summary = {
        "chain_prompt": chain_prompt,
        "landmarks": landmarks,
        "target": target,
        "mental_map": mental_map,
        "found_target": task_success,
        "steps_taken": cnt_step + 1,
        "explore_distance": logger.subtask_explore_dist,
    }
    with open(
        os.path.join(episode_dir, f"mental_map_result_{target_slug}.json"), "w"
    ) as f:
        json.dump(result_summary, f, indent=2)

    logging.info(f"Chain-prompt result: {result_summary}")
    scene.print_scene_graph()
    logging.info("Chain-object-prompt run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="cfg/eval_chain_prompt.yaml", type=str)
    parser.add_argument("--chain_prompt", help="free-text route instruction mentioning landmarks and a final target", required=True, type=str)
    parser.add_argument("--split", help="which episode to use for the scene + start pose", default=1, type=int)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", args.chain_prompt.lower())[:40].strip("_")
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_chainprompt_{slug}_{args.split}.log",
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    logging.info(f"***** Running {cfg.exp_name}: chain prompt *****")
    main(cfg, chain_prompt=args.chain_prompt, split=args.split)
