def compute_cost_matrix(self, trackers: Sequence[Tracklet], detections:
    Sequence[Detection]) ->np.ndarray:
    motion_metric_beta = self.beta
    visual_metric_beta = 1 - self.beta
    if len(trackers) == 0 or len(detections) == 0:
        return np.array([])
    motion_cost = self.motion_metric(trackers, detections)
    motion_cost[motion_cost > self.motion_metric_gate] = np.inf
    visual_cost = self.visual_metric(trackers, detections)
    visual_cost[visual_cost > self.visual_metric_gate] = np.inf
    inf_mask = (motion_cost == np.inf) | (visual_cost == np.inf)
    cost_matrix = (motion_metric_beta * motion_cost + visual_metric_beta *
        visual_cost)
    cost_matrix[inf_mask] = np.inf
    return cost_matrix
