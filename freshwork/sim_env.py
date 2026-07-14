"""Thin habitat-sim wrapper for manual teleoperation.

Scene-loading conventions (mesh/navmesh/semantic file paths, coordinate
transforms) are reused from SCOPE's existing `src/habitat.py` since that is
just simulator plumbing. Everything on top -- the teleop action space, the
step loop, semantic-label lookup at the view center -- is new: SCOPE drives
the agent by teleporting to planned waypoints, this instead steps the agent
one discrete action at a time so a live "mental map" can be built as the
agent actually moves.
"""
import os

import habitat_sim
import numpy as np
import quaternion

from src.habitat import pos_habitat_to_normal


def _yaw_from_rotation(rotation) -> float:
    forward = quaternion.rotate_vectors(rotation, np.array([0.0, 0.0, -1.0]))
    return float(np.arctan2(-forward[0], -forward[2]))


class TeleopSimEnv:
    def __init__(
        self,
        scene_id: str,
        scene_data_path: str,
        scene_dataset_config_path: str,
        img_width: int = 640,
        img_height: int = 480,
        hfov: float = 100,
        sensor_height: float = 1.5,
        camera_tilt_deg: float = 0,
        move_amount: float = 0.25,
        turn_amount_deg: float = 15,
        seed: int = 3407,
    ):
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        split_path = os.path.join(scene_data_path, split)
        stem = scene_id.split("-")[1]
        scene_mesh_path = os.path.join(split_path, scene_id, stem + ".basis.glb")
        navmesh_path = os.path.join(split_path, scene_id, stem + ".basis.navmesh")
        semantic_texture_path = os.path.join(split_path, scene_id, stem + ".semantic.glb")
        self.semantic_annotation_path = os.path.join(split_path, scene_id, stem + ".semantic.txt")

        assert os.path.exists(scene_mesh_path), f"missing scene mesh: {scene_mesh_path}"

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_mesh_path
        sim_cfg.scene_dataset_config_file = scene_dataset_config_path
        sim_cfg.load_semantic_mesh = os.path.exists(semantic_texture_path)

        agent_cfg = habitat_sim.agent.AgentConfiguration()

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [img_height, img_width]
        rgb_spec.position = habitat_sim.geo.UP * sensor_height
        rgb_spec.orientation = [camera_tilt_deg * np.pi / 180, 0.0, 0.0]
        rgb_spec.hfov = hfov
        specs = [rgb_spec]

        if sim_cfg.load_semantic_mesh:
            sem_spec = habitat_sim.CameraSensorSpec()
            sem_spec.uuid = "semantic_sensor"
            sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            sem_spec.resolution = [img_height, img_width]
            sem_spec.position = habitat_sim.geo.UP * sensor_height
            sem_spec.orientation = [camera_tilt_deg * np.pi / 180, 0.0, 0.0]
            sem_spec.hfov = hfov
            specs.append(sem_spec)
        agent_cfg.sensor_specifications = specs

        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=move_amount)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=move_amount)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_amount_deg)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_amount_deg)
            ),
        }

        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self.pathfinder = self.sim.pathfinder
        self.pathfinder.seed(seed)
        if os.path.exists(navmesh_path):
            self.pathfinder.load_nav_mesh(navmesh_path)
        self.agent = self.sim.initialize_agent(0)

        self._place_at_random_start()
        self.instance_to_category = self._load_semantic_annotation()

    def _place_at_random_start(self) -> None:
        state = habitat_sim.AgentState()
        state.position = self.pathfinder.get_random_navigable_point()
        self.agent.set_state(state)

    def _load_semantic_annotation(self) -> dict:
        mapping = {}
        if not os.path.exists(self.semantic_annotation_path):
            return mapping
        with open(self.semantic_annotation_path) as f:
            next(f, None)  # header line
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                try:
                    inst_id = int(parts[0])
                except ValueError:
                    continue
                mapping[inst_id] = parts[2].strip('"')
        return mapping

    def step(self, action: str) -> dict:
        return self.sim.step(action)

    def observations(self) -> dict:
        return self.sim.get_sensor_observations()

    def pose(self):
        """Returns (position_in_normal_frame[xy=horizontal,z=up], yaw, raw_habitat_position)."""
        state = self.agent.get_state()
        pos_normal = pos_habitat_to_normal(np.array(state.position))
        yaw = _yaw_from_rotation(state.rotation)
        return pos_normal, yaw, state.position

    def label_at_center(self, obs: dict):
        if "semantic_sensor" not in obs:
            return None
        sem = obs["semantic_sensor"]
        h, w = sem.shape[:2]
        inst_id = int(sem[h // 2, w // 2])
        return self.instance_to_category.get(inst_id)

    def close(self) -> None:
        self.sim.close()
