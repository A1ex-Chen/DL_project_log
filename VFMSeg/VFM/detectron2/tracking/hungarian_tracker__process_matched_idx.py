def _process_matched_idx(self, instances: Instances, matched_idx: np.
    ndarray, matched_prev_idx: np.ndarray) ->Instances:
    assert matched_idx.size == matched_prev_idx.size
    for i in range(matched_idx.size):
        instances.ID[matched_idx[i]] = self._prev_instances.ID[matched_prev_idx
            [i]]
        instances.ID_period[matched_idx[i]] = self._prev_instances.ID_period[
            matched_prev_idx[i]] + 1
        instances.lost_frame_count[matched_idx[i]] = 0
    return instances
