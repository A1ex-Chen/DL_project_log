def update(self, instances: Instances) ->Instances:
    if instances.has('pred_keypoints'):
        raise NotImplementedError('Need to add support for keypoints')
    instances = self._initialize_extra_fields(instances)
    if self._prev_instances is not None:
        self._untracked_prev_idx = set(range(len(self._prev_instances)))
        cost_matrix = self.build_cost_matrix(instances, self._prev_instances)
        matched_idx, matched_prev_idx = linear_sum_assignment(cost_matrix)
        instances = self._process_matched_idx(instances, matched_idx,
            matched_prev_idx)
        instances = self._process_unmatched_idx(instances, matched_idx)
        instances = self._process_unmatched_prev_idx(instances,
            matched_prev_idx)
    self._prev_instances = copy.deepcopy(instances)
    return instances
