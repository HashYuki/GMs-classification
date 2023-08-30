from typing import List
from . import tools


class Graph:
    """The Graph to model the skeletons extracted by the openpose
    Output shape:
        uniform: (1, num_node, num_node)
        distance*: (1, num_node, num_node)
        distance: (2, num_node, num_node)
        spatial: (3, num_node, num_node)
        DAD: (1, num_node, num_node)
        DLD: (1, num_node, num_node)
    """

    def __init__(
        self, num_node: int, inward: List[int], strategy: str = "uniform"
    ) -> None:
        self.num_node = num_node
        self.self_link = [(i, i) for i in range(num_node)]
        self.inward = inward
        self.outward = [(j, i) for (i, j) in inward]
        self.neighbor = self.inward + self.outward

        self.A = self.get_adjacency_matrix(strategy)

    def get_adjacency_matrix(self, strategy: str):
        if strategy == "uniform":
            A = tools.get_uniform_graph(self.num_node, self.self_link, self.neighbor)
        elif strategy == "distance*":
            A = tools.get_uniform_distance_graph(
                self.num_node, self.self_link, self.neighbor
            )
        elif strategy == "distance":
            A = tools.get_distance_graph(self.num_node, self.self_link, self.neighbor)
        elif strategy == "spatial":
            A = tools.get_spatial_graph(
                self.num_node, self.self_link, self.inward, self.outward
            )
        elif strategy == "DAD":
            A = tools.get_DAD_graph(self.num_node, self.self_link, self.neighbor)
        elif strategy == "DLD":
            A = tools.get_DLD_graph(self.num_node, self.self_link, self.neighbor)
        else:
            raise ValueError()
        return A
