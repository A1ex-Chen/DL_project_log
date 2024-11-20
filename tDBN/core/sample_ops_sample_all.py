def sample_all(self, root_path, gt_boxes, gt_names, num_point_features,
    random_crop=False, gt_group_ids=None, rect=None, Trv2c=None, P2=None):
    sampled_num_dict = {}
    sample_num_per_class = []
    for class_name, max_sample_num in zip(self._sample_classes, self.
        _sample_max_nums):
        sampled_num = int(max_sample_num - np.sum([(n == class_name) for n in
            gt_names]))
        sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
        sampled_num_dict[class_name] = sampled_num
        sample_num_per_class.append(sampled_num)
    sampled_groups = self._sample_classes
    if self._use_group_sampling:
        assert gt_group_ids is not None
        sampled_groups = []
        sample_num_per_class = []
        for group_name, class_names in self._group_name_to_names:
            sampled_nums_group = [sampled_num_dict[n] for n in class_names]
            sampled_num = np.max(sampled_nums_group)
            sample_num_per_class.append(sampled_num)
            sampled_groups.append(group_name)
        total_group_ids = gt_group_ids
    sampled = []
    sampled_gt_boxes = []
    avoid_coll_boxes = gt_boxes
    for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
        if sampled_num > 0:
            if self._use_group_sampling:
                sampled_cls = self.sample_group(class_name, sampled_num,
                    avoid_coll_boxes, total_group_ids)
            else:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                    avoid_coll_boxes)
            sampled += sampled_cls
            if len(sampled_cls) > 0:
                if len(sampled_cls) == 1:
                    sampled_gt_box = sampled_cls[0]['box3d_lidar'][np.
                        newaxis, ...]
                else:
                    sampled_gt_box = np.stack([s['box3d_lidar'] for s in
                        sampled_cls], axis=0)
                sampled_gt_boxes += [sampled_gt_box]
                avoid_coll_boxes = np.concatenate([avoid_coll_boxes,
                    sampled_gt_box], axis=0)
                if self._use_group_sampling:
                    if len(sampled_cls) == 1:
                        sampled_group_ids = np.array(sampled_cls[0]['group_id']
                            )[np.newaxis, ...]
                    else:
                        sampled_group_ids = np.stack([s['group_id'] for s in
                            sampled_cls], axis=0)
                    total_group_ids = np.concatenate([total_group_ids,
                        sampled_group_ids], axis=0)
    if len(sampled) > 0:
        sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
        num_sampled = len(sampled)
        s_points_list = []
        for info in sampled:
            s_points = np.fromfile(str(pathlib.Path(root_path) / info[
                'path']), dtype=np.float32)
            s_points = s_points.reshape([-1, num_point_features])
            if 'rot_transform' in info:
                rot = info['rot_transform']
                s_points[:, :3] = box_np_ops.rotation_points_single_angle(
                    s_points[:, :3], rot, axis=2)
            s_points[:, :3] += info['box3d_lidar'][:3]
            s_points_list.append(s_points)
        if random_crop:
            s_points_list_new = []
            gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect,
                Trv2c, P2)
            crop_frustums = prep.random_crop_frustum(gt_bboxes, rect, Trv2c, P2
                )
            for i in range(crop_frustums.shape[0]):
                s_points = s_points_list[i]
                mask = prep.mask_points_in_corners(s_points, crop_frustums[
                    i:i + 1]).reshape(-1)
                num_remove = np.sum(mask)
                if num_remove > 0 and s_points.shape[0] - num_remove > 15:
                    s_points = s_points[np.logical_not(mask)]
                s_points_list_new.append(s_points)
            s_points_list = s_points_list_new
        ret = {'gt_names': np.array([s['name'] for s in sampled]),
            'difficulty': np.array([s['difficulty'] for s in sampled]),
            'gt_boxes': sampled_gt_boxes, 'points': np.concatenate(
            s_points_list, axis=0), 'gt_masks': np.ones((num_sampled,),
            dtype=np.bool_)}
        if self._use_group_sampling:
            ret['group_ids'] = np.array([s['group_id'] for s in sampled])
        else:
            ret['group_ids'] = np.arange(gt_boxes.shape[0], gt_boxes.shape[
                0] + len(sampled))
    else:
        ret = None
    return ret
