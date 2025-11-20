import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import re

from src.tsdf_planner import Frontier, SnapShot
from src.potential_estimation_gpt_goal import get_potential_estimation


@dataclass
class PotentialNode:
    """A node in the potential graph representing a spatial location."""
    position: np.ndarray  # [x, y] in habitat coordinates
    voxel_position: np.ndarray  # [x, y] in voxel coordinates
    potential_score: float = 0.0  # Current potential score (0-5)
    visit_count: int = 0  # Number of times this area has been explored
    last_updated: int = 0  # Step number when last updated
    frontier_count: int = 0  # Number of frontiers that influenced this node
    exploration_value: float = 0.0  # Derived exploration value
    semantic_richness: float = 0.0  # Average semantic richness
    explorability: float = 0.0  # Average explorability 
    goal_relevance: float = 0.0  # Average goal relevance


class PotentialGraph:
    """
    A spatial graph that tracks exploration potential across the room.
    
    The graph discretizes the navigable space into a grid and maintains
    potential scores for each location based on frontier analysis and
    exploration history.
    """
    
    def __init__(
        self,
        vol_bounds: np.ndarray,  # [x_min, y_min, z_min, x_max, y_max, z_max] or [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        voxel_size: float,
        grid_resolution: float = 0.5,  # Grid spacing in meters
        decay_factor: float = 0.95,  # How fast potential decays over time
        influence_radius: float = 2.0,  # How far frontier potential spreads
    ):
        self.vol_bounds = vol_bounds
        self.voxel_size = voxel_size
        self.grid_resolution = grid_resolution
        self.decay_factor = decay_factor
        self.influence_radius = influence_radius
        
        # Handle different vol_bounds formats
        if vol_bounds.shape == (3, 2):
            # Format: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            self.x_min, self.x_max = vol_bounds[0, 0], vol_bounds[0, 1]
            self.y_min, self.y_max = vol_bounds[1, 0], vol_bounds[1, 1]
        elif vol_bounds.shape == (6,):
            # Format: [x_min, y_min, z_min, x_max, y_max, z_max]
            self.x_min, self.y_min = vol_bounds[0], vol_bounds[1]
            self.x_max, self.y_max = vol_bounds[3], vol_bounds[4]
        else:
            raise ValueError(f"Unsupported vol_bounds shape: {vol_bounds.shape}. Expected (3, 2) or (6,)")
        
        # Grid dimensions
        self.grid_width = int((self.x_max - self.x_min) / grid_resolution) + 1
        self.grid_height = int((self.y_max - self.y_min) / grid_resolution) + 1
        
        # Initialize nodes
        self.nodes: Dict[Tuple[int, int], PotentialNode] = {}
        self._initialize_grid()
        
        # Tracking
        self.current_step = 0
        self.update_history = []  # Track updates for analysis
        
        logging.info(f"Initialized PotentialGraph with {len(self.nodes)} nodes "
                    f"({self.grid_width}x{self.grid_height} grid)")
    
    def _initialize_grid(self):
        """Initialize the spatial grid with nodes."""
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # Convert grid indices to world coordinates
                x = self.x_min + i * self.grid_resolution
                y = self.y_min + j * self.grid_resolution
                
                # Convert to voxel coordinates - handle different vol_bounds formats
                if self.vol_bounds.shape == (3, 2):
                    voxel_x = int((x - self.vol_bounds[0, 0]) / self.voxel_size)
                    voxel_y = int((y - self.vol_bounds[1, 0]) / self.voxel_size)
                else:
                    voxel_x = int((x - self.vol_bounds[0]) / self.voxel_size)
                    voxel_y = int((y - self.vol_bounds[1]) / self.voxel_size)
                
                node = PotentialNode(
                    position=np.array([x, y]),
                    voxel_position=np.array([voxel_x, voxel_y])
                )
                self.nodes[(i, j)] = node
    
    def world_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        # Fixed: Ensure consistent coordinate handling
        # position should be [x, z] for horizontal plane navigation
        if len(position) == 3:
            # For 3D position [x, y, z], use x and z (y is height in Habitat)
            x, z = position[0], position[2]
        elif len(position) == 2:
            # For 2D position, assume [x, z] format
            x, z = position[0], position[1]
        else:
            raise ValueError(f"Position must be 2D or 3D, got shape {position.shape}")
        
        i = int((x - self.x_min) / self.grid_resolution)
        j = int((z - self.y_min) / self.grid_resolution)  # y_min/y_max represent z bounds
        
        # Clamp to valid range
        i = max(0, min(i, self.grid_width - 1))
        j = max(0, min(j, self.grid_height - 1))
        
        return (i, j)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        i, j = grid_pos
        x = self.x_min + i * self.grid_resolution
        y = self.y_min + j * self.grid_resolution
        return np.array([x, y])
    
    def update_from_frontier(
        self,
        frontier: Frontier,
        subtask_metadata: dict,
        occupied_map: np.ndarray = None,
        potential_text: str = None
    ):
        """Update potential scores based on a frontier's analysis."""
        self.current_step += 1
        
        # Use pre-computed potential text if provided, otherwise compute it
        if potential_text is not None:
            potential_scores = self._parse_potential_text(potential_text)
        elif frontier.feature is not None:
            potential_text = get_potential_estimation(subtask_metadata, frontier.feature)
            potential_scores = self._parse_potential_text(potential_text)
        else:
            # Default scores if no analysis available
            potential_scores = {
                'semantic_richness': 3.0,  # Use 3.0 as neutral default
                'explorability': 3.0, 
                'goal_relevance': 3.0,
                'potential_score': 3.0
            }
        
        # Get frontier position in world coordinates - Fixed coordinate conversion
        frontier_world_pos = self._voxel_to_world(frontier.position)
        
        # Update nodes within influence radius
        updated_nodes = []
        for (i, j), node in self.nodes.items():
            # Fixed: Use consistent 2D coordinates for distance calculation
            if len(frontier_world_pos) == 3:
                # Extract x, z coordinates for horizontal distance
                frontier_pos_2d = np.array([frontier_world_pos[0], frontier_world_pos[2]])
            else:
                frontier_pos_2d = frontier_world_pos[:2]
            
            distance = np.linalg.norm(node.position - frontier_pos_2d)
            
            if distance <= self.influence_radius:
                # Calculate influence weight (closer = more influence)
                weight = max(0, 1.0 - distance / self.influence_radius)
                
                # Skip if area is occupied (optional)
                if occupied_map is not None:
                    voxel_pos = node.voxel_position
                    if (0 <= voxel_pos[0] < occupied_map.shape[0] and 
                        0 <= voxel_pos[1] < occupied_map.shape[1] and
                        occupied_map[voxel_pos[0], voxel_pos[1]]):
                        continue
                
                # Update node with weighted scores
                self._update_node_scores(node, potential_scores, weight)
                updated_nodes.append((i, j))
        
        # Log update
        update_info = {
            'step': self.current_step,
            'frontier_pos': frontier_pos_2d.tolist(),
            'scores': potential_scores,
            'updated_nodes': len(updated_nodes)
        }
        self.update_history.append(update_info)
        
        logging.debug(f"Updated {len(updated_nodes)} nodes from frontier at {frontier_pos_2d} with scores {potential_scores}")
    
    def _parse_potential_text(self, potential_text: str) -> Dict[str, float]:
        """Parse the GPT potential estimation text to extract numerical scores."""
        scores = {
            'semantic_richness': 3.0,  # Increase default from 2.0 to 3.0
            'explorability': 3.0,
            'goal_relevance': 3.0, 
            'potential_score': 3.0
        }
        
        if not potential_text:
            return scores
        
        try:
            # Convert to lowercase for matching
            text_lower = potential_text.lower()
            
            # Parse semantic richness with more robust pattern matching
            if '**semantic_richness:**' in text_lower:
                start_idx = text_lower.find('**semantic_richness:**') + len('**semantic_richness:**')
                end_idx = text_lower.find('\n', start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 50  # Fallback
                richness_text = text_lower[start_idx:end_idx].strip()
                
                if 'high' in richness_text:
                    scores['semantic_richness'] = 4.5
                elif 'medium' in richness_text:
                    scores['semantic_richness'] = 3.0
                elif 'low' in richness_text:
                    scores['semantic_richness'] = 1.5
            
            # Parse explorability
            if '**explorability:**' in text_lower:
                start_idx = text_lower.find('**explorability:**') + len('**explorability:**')
                end_idx = text_lower.find('\n', start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 50
                explorability_text = text_lower[start_idx:end_idx].strip()
                
                if 'high' in explorability_text:
                    scores['explorability'] = 4.5
                elif 'medium' in explorability_text:
                    scores['explorability'] = 3.0
                elif 'low' in explorability_text:
                    scores['explorability'] = 1.5
            
            # Parse goal relevance
            if '**goal_relevance:**' in text_lower:
                start_idx = text_lower.find('**goal_relevance:**') + len('**goal_relevance:**')
                end_idx = text_lower.find('\n', start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 50
                relevance_text = text_lower[start_idx:end_idx].strip()
                
                if 'high' in relevance_text:
                    scores['goal_relevance'] = 4.5
                elif 'medium' in relevance_text:
                    scores['goal_relevance'] = 3.0
                elif 'low' in relevance_text:
                    scores['goal_relevance'] = 1.5
            
            # Look for the structured potential score first
            potential_match = re.search(r'\*\*potential_score:\*\*\s*(\d+\.?\d*)', text_lower)
            if potential_match:
                scores['potential_score'] = float(potential_match.group(1))
            else:
                # Fallback to any score pattern
                score_patterns = [
                    r'potential score.*?(\d+\.?\d*)',
                    r'score.*?(\d+\.?\d*)',
                    r'rating.*?(\d+\.?\d*)'
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        potential_score = float(match.group(1))
                        # Clamp to valid range
                        scores['potential_score'] = max(1.0, min(5.0, potential_score))
                        break
                        
        except Exception as e:
            print(f"Error parsing potential text: {e}")
            print(f"Text was: {potential_text}")
        
        # Log parsed scores for debugging
        print(f"Parsed scores: {scores}")
        return scores
    
    def _update_node_scores(self, node: PotentialNode, scores: Dict[str, float], weight: float):
        """Update a node's scores with weighted averaging."""
        # Decay existing scores based on time
        steps_since_update = self.current_step - node.last_updated
        if steps_since_update > 0:
            decay = self.decay_factor ** steps_since_update
            node.potential_score *= decay
            node.semantic_richness *= decay
            node.explorability *= decay
            node.goal_relevance *= decay
        
        # Use a more aggressive learning rate for new information
        # Especially for the first few frontiers
        if node.frontier_count == 0:
            alpha = weight  # Full weight for first update
        elif node.frontier_count < 3:
            alpha = weight * 0.7  # High weight for early updates
        else:
            alpha = weight * 0.5  # More conservative for later updates
        
        # Update with new weighted scores
        node.semantic_richness = (1 - alpha) * node.semantic_richness + alpha * scores['semantic_richness']
        node.explorability = (1 - alpha) * node.explorability + alpha * scores['explorability']
        node.goal_relevance = (1 - alpha) * node.goal_relevance + alpha * scores['goal_relevance']
        node.potential_score = (1 - alpha) * node.potential_score + alpha * scores['potential_score']
        
        # Update metadata
        node.frontier_count += 1
        node.last_updated = self.current_step
        
        # Calculate exploration value with higher weights for good scores
        base_exploration = (
            node.potential_score * 0.5 +  # Give more weight to overall score
            node.explorability * 0.3 + 
            node.goal_relevance * 0.2
        )
        
        # Apply diminishing returns for visited areas, but less aggressive
        visit_penalty = max(0.1, 1.0 / (1 + node.visit_count * 0.5))
        node.exploration_value = base_exploration * visit_penalty
    
    def mark_visited(self, position: np.ndarray, radius: float = 1.0):
        """Mark an area as visited, reducing its exploration value."""
        # Fixed: Ensure consistent coordinate handling
        if len(position) == 3:
            # For 3D Habitat position [x, y, z], use x, z for horizontal plane
            pos_2d = np.array([position[0], position[2]])
        elif len(position) == 2:
            # Assume already in correct 2D format
            pos_2d = position
        else:
            raise ValueError(f"Position must be 2D or 3D, got shape {position.shape}")
        
        # Mark nearby nodes as visited
        for (i, j), node in self.nodes.items():
            distance = np.linalg.norm(node.position - pos_2d)
            if distance <= radius:
                node.visit_count += 1
                # Recalculate exploration value with proper formula
                base_value = (
                    node.potential_score * 0.4 +
                    node.explorability * 0.3 + 
                    node.goal_relevance * 0.3
                )
                visit_penalty = max(0.1, 1.0 / (1 + node.visit_count * 0.5))
                node.exploration_value = base_value * visit_penalty
    
    def get_highest_potential_positions(self, n: int = 5, min_distance: float = 1.0) -> List[Tuple[np.ndarray, float]]:
        """Get the N positions with highest exploration potential."""
        # Sort nodes by exploration value
        sorted_nodes = sorted(
            [(pos, node) for pos, node in self.nodes.items()],
            key=lambda x: x[1].exploration_value,
            reverse=True
        )
        
        selected = []
        for grid_pos, node in sorted_nodes:
            world_pos = node.position
            
            # Check minimum distance from already selected positions
            too_close = False
            for selected_pos, _ in selected:
                if np.linalg.norm(world_pos - selected_pos) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected.append((world_pos, node.exploration_value))
                if len(selected) >= n:
                    break
        
        return selected
    
    def get_potential_at_position(self, position: np.ndarray) -> float:
        """Get the potential score at a specific position."""
        # Handle both 2D and 3D position arrays consistently
        if len(position) >= 3:
            # For 3D Habitat coordinates [x, y, z], use x and z for horizontal plane (y is height)
            pos_2d = np.array([position[0], position[2]])  # x, z are horizontal
        else:
            # For 2D coordinates, assume they are already [x, z] horizontal
            pos_2d = position[:2]
            
        grid_pos = self.world_to_grid(pos_2d)
        if grid_pos in self.nodes:
            return self.nodes[grid_pos].potential_score
        return 3.0  # Return meaningful default instead of 0.0
    
    def _voxel_to_world(self, voxel_pos: np.ndarray) -> np.ndarray:
        """Convert voxel coordinates to world coordinates."""
        # Fixed: Handle different vol_bounds formats consistently
        if self.vol_bounds.shape == (3, 2):
            world_x = voxel_pos[0] * self.voxel_size + self.vol_bounds[0, 0]
            world_z = voxel_pos[1] * self.voxel_size + self.vol_bounds[1, 0]
            world_y = (self.vol_bounds[2, 0] + self.vol_bounds[2, 1]) / 2  # Use middle of z range for height
        else:
            world_x = voxel_pos[0] * self.voxel_size + self.vol_bounds[0]
            world_z = voxel_pos[1] * self.voxel_size + self.vol_bounds[1]
            world_y = self.vol_bounds[2]
        
        return np.array([world_x, world_y, world_z])
    
    def visualize(self, save_path: Optional[str] = None, title: str = "Potential Graph"):
        """Visualize the current potential landscape."""
        # Create potential map
        potential_map = np.zeros((self.grid_height, self.grid_width))
        exploration_map = np.zeros((self.grid_height, self.grid_width))
        visit_map = np.zeros((self.grid_height, self.grid_width))
        
        for (i, j), node in self.nodes.items():
            potential_map[j, i] = node.potential_score
            exploration_map[j, i] = node.exploration_value
            visit_map[j, i] = min(node.visit_count, 10)  # Cap for visualization
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)
        
        # Potential scores
        im1 = axes[0, 0].imshow(potential_map, cmap='viridis', origin='lower')
        axes[0, 0].set_title('Potential Scores')
        axes[0, 0].set_xlabel('Grid X')
        axes[0, 0].set_ylabel('Grid Y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Exploration values
        im2 = axes[0, 1].imshow(exploration_map, cmap='plasma', origin='lower')
        axes[0, 1].set_title('Exploration Values')
        axes[0, 1].set_xlabel('Grid X')
        axes[0, 1].set_ylabel('Grid Y')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Visit counts
        im3 = axes[1, 0].imshow(visit_map, cmap='Blues', origin='lower')
        axes[1, 0].set_title('Visit Counts')
        axes[1, 0].set_xlabel('Grid X')
        axes[1, 0].set_ylabel('Grid Y')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Smoothed potential (for planning)
        smoothed_potential = gaussian_filter(potential_map, sigma=1.0)
        im4 = axes[1, 1].imshow(smoothed_potential, cmap='viridis', origin='lower')
        axes[1, 1].set_title('Smoothed Potential')
        axes[1, 1].set_xlabel('Grid X')
        axes[1, 1].set_ylabel('Grid Y')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logging.debug(f"Saved potential graph visualization to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def save_state(self, filepath: str):
        """Save the graph state to disk."""
        state = {
            'nodes': self.nodes,
            'current_step': self.current_step,
            'update_history': self.update_history,
            'config': {
                'vol_bounds': self.vol_bounds,
                'voxel_size': self.voxel_size,
                'grid_resolution': self.grid_resolution,
                'decay_factor': self.decay_factor,
                'influence_radius': self.influence_radius
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logging.info(f"Saved potential graph state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the graph state from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.nodes = state['nodes']
        self.current_step = state['current_step']
        self.update_history = state['update_history']
        
        logging.info(f"Loaded potential graph state from {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the current graph state."""
        if not self.nodes:
            return {}
        
        potential_scores = [node.potential_score for node in self.nodes.values()]
        exploration_values = [node.exploration_value for node in self.nodes.values()]
        visit_counts = [node.visit_count for node in self.nodes.values()]
        
        stats = {
            'total_nodes': len(self.nodes),
            'avg_potential': np.mean(potential_scores),
            'max_potential': np.max(potential_scores),
            'avg_exploration_value': np.mean(exploration_values),
            'max_exploration_value': np.max(exploration_values),
            'total_visits': sum(visit_counts),
            'nodes_visited': sum(1 for count in visit_counts if count > 0),
            'coverage_ratio': sum(1 for count in visit_counts if count > 0) / len(self.nodes),
            'current_step': self.current_step
        }
        
        return stats
