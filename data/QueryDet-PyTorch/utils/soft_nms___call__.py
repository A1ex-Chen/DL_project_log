def __call__(self, boxes, scores, class_idxs):
    return batched_soft_nms(boxes, scores, class_idxs, self.method, self.
        gaussian_sigma, self.linear_threshold, self.prune_threshold)
