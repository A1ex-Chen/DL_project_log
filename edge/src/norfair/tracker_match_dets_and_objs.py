def match_dets_and_objs(self, distance_matrix: np.ndarray, distance_threshold):
    """Matches detections with tracked_objects from a distance matrix

        I used to match by minimizing the global distances, but found several
        cases in which this was not optimal. So now I just match by starting
        with the global minimum distance and matching the det-obj corresponding
        to that distance, then taking the second minimum, and so on until we
        reach the distance_threshold.

        This avoids the the algorithm getting cute with us and matching things
        that shouldn't be matching just for the sake of minimizing the global
        distance, which is what used to happen
        """
    distance_matrix = distance_matrix.copy()
    if distance_matrix.size > 0:
        det_idxs = []
        obj_idxs = []
        current_min = distance_matrix.min()
        while current_min < distance_threshold:
            flattened_arg_min = distance_matrix.argmin()
            det_idx = flattened_arg_min // distance_matrix.shape[1]
            obj_idx = flattened_arg_min % distance_matrix.shape[1]
            det_idxs.append(det_idx)
            obj_idxs.append(obj_idx)
            distance_matrix[det_idx, :] = distance_threshold + 1
            distance_matrix[:, obj_idx] = distance_threshold + 1
            current_min = distance_matrix.min()
        return det_idxs, obj_idxs
    else:
        return [], []
