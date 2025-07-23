"""
Methods to quantify morphological features of venation patterns.
"""

import numpy as np
import sknw
from networkx import Graph
from skimage.morphology import thin, skeletonize

MIN_NODE_DEGREE = 2


class VenationMorphometer:
    def __init__(self):
        """
        Initialize the VenationMorphometer.
        """
        pass

    def _thin_and_skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """
        """
        _thin_mask = thin(mask)
        return skeletonize(_thin_mask > 0)        

    def run(self, binary_mask: np.ndarray, prune: bool = True) -> Graph:
        """
        Computes the skeleton graph from a binary mask.

        Args:
            binary_mask (np.ndarray): Binary mask image.
            prune (bool): Whether to prune the graph by removing nodes with degree less than MIN_NODE_DEGREE.

        Returns:
            nx.Graph: Graph representation of the skeleton.
        """
        if binary_mask.ndim != 2:
            raise ValueError(f"Expect 2D binary mask but got {binary_mask.ndim}")

        if binary_mask.dtype != bool:
            raise ValueError("Binary mask must be a binary image (dtype should be bool).")

        skeleton = self._thin_and_skeletonize(binary_mask)

        # Convert the binary skeleton to a graph
        graph = sknw.build_sknw(skeleton)

        if prune:
            # Remove nodes with degree less than MIN_NODE_DEGREE
            remove = [
                node for node, degree in graph.degree() if degree < MIN_NODE_DEGREE
            ]
            graph.remove_nodes_from(remove)

        self.graph = graph
        return self.graph
