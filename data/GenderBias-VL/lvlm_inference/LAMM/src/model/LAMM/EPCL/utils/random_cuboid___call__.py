def __call__(self, point_cloud, target_boxes, per_point_labels=None):
    range_xyz = np.max(point_cloud[:, 0:3], axis=0) - np.min(point_cloud[:,
        0:3], axis=0)
    for _ in range(100):
        crop_range = self.min_crop + np.random.rand(3) * (self.max_crop -
            self.min_crop)
        if not check_aspect(crop_range, self.aspect):
            continue
        sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]
        new_range = range_xyz * crop_range / 2.0
        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range
        upper_idx = np.sum((point_cloud[:, 0:3] <= max_xyz).astype(np.int32), 1
            ) == 3
        lower_idx = np.sum((point_cloud[:, 0:3] >= min_xyz).astype(np.int32), 1
            ) == 3
        new_pointidx = upper_idx & lower_idx
        if np.sum(new_pointidx) < self.min_points:
            continue
        new_point_cloud = point_cloud[new_pointidx, :]
        if self.box_filter_policy == 'center':
            new_boxes = target_boxes
            if target_boxes.sum() > 0:
                box_centers = target_boxes[:, 0:3]
                new_pc_min_max = np.min(new_point_cloud[:, 0:3], axis=0
                    ), np.max(new_point_cloud[:, 0:3], axis=0)
                keep_boxes = np.logical_and(np.all(box_centers >=
                    new_pc_min_max[0], axis=1), np.all(box_centers <=
                    new_pc_min_max[1], axis=1))
                if keep_boxes.sum() == 0:
                    continue
                new_boxes = target_boxes[keep_boxes]
            if per_point_labels is not None:
                new_per_point_labels = [x[new_pointidx] for x in
                    per_point_labels]
            else:
                new_per_point_labels = None
            return new_point_cloud, new_boxes, new_per_point_labels
    return point_cloud, target_boxes, per_point_labels
