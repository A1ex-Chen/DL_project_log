def _process_unmatched_idx(self, instances: Instances, matched_idx: np.ndarray
    ) ->Instances:
    untracked_idx = set(range(len(instances))).difference(set(matched_idx))
    for idx in untracked_idx:
        instances.ID[idx] = self._id_count
        self._id_count += 1
        instances.ID_period[idx] = 1
        instances.lost_frame_count[idx] = 0
    return instances
