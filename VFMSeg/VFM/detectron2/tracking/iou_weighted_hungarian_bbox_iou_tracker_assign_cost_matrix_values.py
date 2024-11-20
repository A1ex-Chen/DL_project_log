def assign_cost_matrix_values(self, cost_matrix: np.ndarray, bbox_pairs: List
    ) ->np.ndarray:
    """
        Based on IoU for each pair of bbox, assign the associated value in cost matrix

        Args:
            cost_matrix: np.ndarray, initialized 2D array with target dimensions
            bbox_pairs: list of bbox pair, in each pair, iou value is stored
        Return:
            np.ndarray, cost_matrix with assigned values
        """
    for pair in bbox_pairs:
        cost_matrix[pair['idx']][pair['prev_idx']] = -1 * pair['IoU']
    return cost_matrix
