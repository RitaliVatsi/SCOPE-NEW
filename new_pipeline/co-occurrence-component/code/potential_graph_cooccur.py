"""
Co-occurrence-aware PotentialGraph (Gap 2 in ../../GAPS.md).

Subclasses src/potential_graph.py's PotentialGraph rather than editing it, so
run_object_goal.py / run_goatbench_evaluation.py (which import the original
PotentialGraph directly) are completely unaffected. Only
chain_object_prompt_cooccur.py imports this variant.

What changes vs. the base class:
- Nodes gain a `room_plausibility` score, parsed from a new **ROOM_PLAUSIBILITY**
  field in the GPT-4o response (see potential_estimation_cooccur.py's prompt).
- room_plausibility is a semantic co-occurrence prior ("does this frontier's
  room type typically contain the goal object"), not spatial proximity to a
  previously-found landmark - GAPS.md's empirical evidence is that chains jump
  between unrelated rooms, so proximity-based boosting would be wrong.
- It down-weights (not zeroes) exploration_value, so a frontier in an
  implausible room type can still be surfaced if nothing better is available,
  but a plausible one is preferred - matching "down-weight before a candidate
  is surfaced" rather than a hard filter.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.potential_graph import PotentialGraph, PotentialNode
from potential_estimation_cooccur import get_potential_estimation


@dataclass
class PotentialNodeCooccur(PotentialNode):
    room_plausibility: float = 0.0  # semantic object/room co-occurrence prior (0-5)


def _plausibility_factor(room_plausibility: float) -> float:
    """Map a 1.0-5.0 room_plausibility score to a soft down-weighting factor in [0.3, 1.0].

    Deliberately never reaches 0 - this is a soft commonsense prior, not a hard
    filter, since the VLM's room-type read can itself be wrong.
    """
    clamped = max(1.0, min(5.0, room_plausibility))
    return 0.3 + 0.7 * (clamped - 1.0) / 4.0


class PotentialGraphCooccur(PotentialGraph):
    """PotentialGraph with a chain-aware, co-occurrence room-plausibility term."""

    def _initialize_grid(self):
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = self.x_min + i * self.grid_resolution
                y = self.y_min + j * self.grid_resolution

                if self.vol_bounds.shape == (3, 2):
                    voxel_x = int((x - self.vol_bounds[0, 0]) / self.voxel_size)
                    voxel_y = int((y - self.vol_bounds[1, 0]) / self.voxel_size)
                else:
                    voxel_x = int((x - self.vol_bounds[0]) / self.voxel_size)
                    voxel_y = int((y - self.vol_bounds[1]) / self.voxel_size)

                self.nodes[(i, j)] = PotentialNodeCooccur(
                    position=np.array([x, y]),
                    voxel_position=np.array([voxel_x, voxel_y]),
                )

    def update_from_frontier(
        self,
        frontier,
        subtask_metadata: dict,
        occupied_map=None,
        potential_text: str = None,
    ):
        """Same as base class, but calls the co-occurrence-aware estimator."""
        self.current_step += 1

        if potential_text is not None:
            potential_scores = self._parse_potential_text(potential_text)
        elif frontier.feature is not None:
            potential_text = get_potential_estimation(subtask_metadata, frontier.feature)
            potential_scores = self._parse_potential_text(potential_text)
        else:
            potential_scores = {
                'semantic_richness': 3.0,
                'explorability': 3.0,
                'goal_relevance': 3.0,
                'room_plausibility': 3.0,
                'potential_score': 3.0,
            }

        frontier_world_pos = self._voxel_to_world(frontier.position)

        updated_nodes = []
        for (i, j), node in self.nodes.items():
            if len(frontier_world_pos) == 3:
                frontier_pos_2d = np.array([frontier_world_pos[0], frontier_world_pos[2]])
            else:
                frontier_pos_2d = frontier_world_pos[:2]

            distance = np.linalg.norm(node.position - frontier_pos_2d)

            if distance <= self.influence_radius:
                weight = max(0, 1.0 - distance / self.influence_radius)

                if occupied_map is not None:
                    voxel_pos = node.voxel_position
                    if (0 <= voxel_pos[0] < occupied_map.shape[0] and
                            0 <= voxel_pos[1] < occupied_map.shape[1] and
                            occupied_map[voxel_pos[0], voxel_pos[1]]):
                        continue

                self._update_node_scores(node, potential_scores, weight)
                updated_nodes.append((i, j))

        update_info = {
            'step': self.current_step,
            'frontier_pos': frontier_pos_2d.tolist(),
            'scores': potential_scores,
            'updated_nodes': len(updated_nodes),
        }
        self.update_history.append(update_info)

        logging.debug(
            f"Updated {len(updated_nodes)} nodes from frontier at {frontier_pos_2d} "
            f"with scores {potential_scores}"
        )

    def _parse_potential_text(self, potential_text: str) -> Dict[str, float]:
        scores = super()._parse_potential_text(potential_text)
        scores.setdefault('room_plausibility', 3.0)

        if not potential_text:
            return scores

        try:
            text_lower = potential_text.lower()
            if '**room_plausibility:**' in text_lower:
                start_idx = text_lower.find('**room_plausibility:**') + len('**room_plausibility:**')
                end_idx = text_lower.find('\n', start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 50
                plausibility_text = text_lower[start_idx:end_idx].strip()

                if 'high' in plausibility_text:
                    scores['room_plausibility'] = 4.5
                elif 'medium' in plausibility_text:
                    scores['room_plausibility'] = 3.0
                elif 'low' in plausibility_text:
                    scores['room_plausibility'] = 1.5
        except Exception as e:
            print(f"Error parsing room_plausibility: {e}")

        print(f"Parsed scores (with room_plausibility): {scores}")
        return scores

    def _update_node_scores(self, node: PotentialNodeCooccur, scores: Dict[str, float], weight: float):
        steps_since_update = self.current_step - node.last_updated
        if steps_since_update > 0:
            decay = self.decay_factor ** steps_since_update
            node.potential_score *= decay
            node.semantic_richness *= decay
            node.explorability *= decay
            node.goal_relevance *= decay
            node.room_plausibility *= decay

        if node.frontier_count == 0:
            alpha = weight
        elif node.frontier_count < 3:
            alpha = weight * 0.7
        else:
            alpha = weight * 0.5

        node.semantic_richness = (1 - alpha) * node.semantic_richness + alpha * scores['semantic_richness']
        node.explorability = (1 - alpha) * node.explorability + alpha * scores['explorability']
        node.goal_relevance = (1 - alpha) * node.goal_relevance + alpha * scores['goal_relevance']
        node.potential_score = (1 - alpha) * node.potential_score + alpha * scores['potential_score']
        node.room_plausibility = (1 - alpha) * node.room_plausibility + alpha * scores['room_plausibility']

        node.frontier_count += 1
        node.last_updated = self.current_step

        base_exploration = (
            node.potential_score * 0.5 +
            node.explorability * 0.3 +
            node.goal_relevance * 0.2
        )

        # Co-occurrence gating: down-weight (never zero) exploration_value for
        # frontiers in a semantically implausible room type, applied before the
        # candidate is ranked/surfaced by get_highest_potential_positions().
        plausibility_factor = _plausibility_factor(node.room_plausibility)

        visit_penalty = max(0.1, 1.0 / (1 + node.visit_count * 0.5))
        node.exploration_value = base_exploration * plausibility_factor * visit_penalty

    def mark_visited(self, position, radius: float = 1.0):
        if len(position) == 3:
            pos_2d = np.array([position[0], position[2]])
        elif len(position) == 2:
            pos_2d = position
        else:
            raise ValueError(f"Position must be 2D or 3D, got shape {position.shape}")

        for (i, j), node in self.nodes.items():
            distance = np.linalg.norm(node.position - pos_2d)
            if distance <= radius:
                node.visit_count += 1
                base_value = (
                    node.potential_score * 0.4 +
                    node.explorability * 0.3 +
                    node.goal_relevance * 0.3
                )
                plausibility_factor = _plausibility_factor(node.room_plausibility)
                visit_penalty = max(0.1, 1.0 / (1 + node.visit_count * 0.5))
                node.exploration_value = base_value * plausibility_factor * visit_penalty
