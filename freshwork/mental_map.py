"""Incrementally-built map of a scene the agent has no prior knowledge of.

As the agent moves, it appends its pose to a trajectory graph. Landmarks are
marked live (at whatever pose the agent is at when marked) and must be
reached in the order they were marked -- `check_arrival` only tests against
the current *active* landmark, so "go to landmark 2" implicitly requires
landmark 1 to already be reached.
"""
import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PoseNode:
    id: int
    position: List[float]  # [x, y, z], z is up
    heading: float  # yaw, radians
    t: float


@dataclass
class Landmark:
    id: int
    label: str
    position: List[float]
    radius: float
    reached: bool = False
    reached_at_node: Optional[int] = None


class MentalMap:
    def __init__(self, arrival_radius: float = 1.0):
        self.nodes: List[PoseNode] = []
        self.edges: List[Tuple[int, int]] = []
        self.landmarks: List[Landmark] = []
        self.arrival_radius = arrival_radius
        self._active_landmark_idx = 0

    def observe_pose(self, position, heading) -> PoseNode:
        node = PoseNode(
            id=len(self.nodes),
            position=[float(v) for v in position],
            heading=float(heading),
            t=time.time(),
        )
        if self.nodes:
            self.edges.append((self.nodes[-1].id, node.id))
        self.nodes.append(node)
        return node

    def add_landmark(self, label: str, position, radius: Optional[float] = None) -> Landmark:
        lm = Landmark(
            id=len(self.landmarks),
            label=label,
            position=[float(v) for v in position],
            radius=radius if radius is not None else self.arrival_radius,
        )
        self.landmarks.append(lm)
        return lm

    @property
    def active_landmark(self) -> Optional[Landmark]:
        if self._active_landmark_idx >= len(self.landmarks):
            return None
        return self.landmarks[self._active_landmark_idx]

    def check_arrival(self, position) -> Optional[Landmark]:
        """Test the current position against the active landmark only.
        Returns the landmark if it was just reached, else None."""
        lm = self.active_landmark
        if lm is None:
            return None
        dist = float(np.linalg.norm(np.array(position) - np.array(lm.position)))
        if dist <= lm.radius:
            lm.reached = True
            lm.reached_at_node = self.nodes[-1].id if self.nodes else None
            self._active_landmark_idx += 1
            return lm
        return None

    def distance_to_active(self, position) -> Optional[float]:
        lm = self.active_landmark
        if lm is None:
            return None
        return float(np.linalg.norm(np.array(position) - np.array(lm.position)))

    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": self.edges,
            "landmarks": [asdict(l) for l in self.landmarks],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
