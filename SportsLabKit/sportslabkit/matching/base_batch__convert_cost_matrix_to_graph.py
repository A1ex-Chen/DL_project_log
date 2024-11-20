@abstractmethod
def _convert_cost_matrix_to_graph(self, cost_matrices: list[np.ndarray],
    no_detection_cost: float=100000.0) ->tuple[list[int], list[int], list[
    int], list[int], list[int], dict[int, Node], int, int]:
    """Transforms cost matrix into graph representation for min cost flow computation.

        Args:
            cost_matricies: A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame k.

        Returns:
            A tuple containing arrays of start nodes, end nodes, capacities, unit costs, and supplies.
        """
    pass
