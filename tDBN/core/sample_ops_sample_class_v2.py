def sample_class_v2(self, name, num, gt_boxes):
    sampled = self._sampler_dict[name].sample(num)
    sampled = copy.deepcopy(sampled)
    num_gt = gt_boxes.shape[0]
    num_sampled = len(sampled)
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(gt_boxes[:, 0:2],
        gt_boxes[:, 3:5], gt_boxes[:, 6])
    sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
    valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
    valid_mask = np.concatenate([valid_mask, np.ones([sp_boxes.shape[0]],
        dtype=np.bool_)], axis=0)
    boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
    if self._enable_global_rot:
        prep.noise_per_object_v3_(boxes, None, valid_mask, 0, 0, self.
            _global_rot_range, num_try=100)
    sp_boxes_new = boxes[gt_boxes.shape[0]:]
    sp_boxes_bv = box_np_ops.center_to_corner_box2d(sp_boxes_new[:, 0:2],
        sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])
    total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
    coll_mat = prep.box_collision_test(total_bv, total_bv)
    diag = np.arange(total_bv.shape[0])
    coll_mat[diag, diag] = False
    valid_samples = []
    for i in range(num_gt, num_gt + num_sampled):
        if coll_mat[i].any():
            coll_mat[i] = False
            coll_mat[:, i] = False
        else:
            if self._enable_global_rot:
                sampled[i - num_gt]['box3d_lidar'][:2] = boxes[i, :2]
                sampled[i - num_gt]['box3d_lidar'][-1] = boxes[i, -1]
                sampled[i - num_gt]['rot_transform'] = boxes[i, -1] - sp_boxes[
                    i - num_gt, -1]
            valid_samples.append(sampled[i - num_gt])
    return valid_samples
