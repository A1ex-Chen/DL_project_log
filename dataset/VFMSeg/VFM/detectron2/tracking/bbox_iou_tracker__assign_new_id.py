def _assign_new_id(self, instances: Instances) ->Instances:
    """
        For each untracked instance, assign a new id

        Args:
            instances: D2 Instances, for predictions of the current frame
        Return:
            D2 Instances with new ID assigned
        """
    untracked_idx = set(range(len(instances))).difference(self._matched_idx)
    for idx in untracked_idx:
        instances.ID[idx] = self._id_count
        self._id_count += 1
        instances.ID_period[idx] = 1
        instances.lost_frame_count[idx] = 0
    return instances
