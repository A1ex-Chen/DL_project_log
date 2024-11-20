def sample_group(self, name, num, gt_boxes, gt_group_ids):
    sampled, group_num = self.sample(name, num)
    sampled = copy.deepcopy(sampled)
    gid_map = {}
    max_gt_gid = np.max(gt_group_ids)
    sampled_gid = max_gt_gid + 1
    for s in sampled:
        gid = s['group_id']
        if gid in gid_map:
            s['group_id'] = gid_map[gid]
        else:
            gid_map[gid] = sampled_gid
            s['group_id'] = sampled_gid
            sampled_gid += 1
    num_gt = gt_boxes.shape[0]
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(gt_boxes[:, 0:2],
        gt_boxes[:, 3:5], gt_boxes[:, 6])
    sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
    sp_group_ids = np.stack([i['group_id'] for i in sampled], axis=0)
    valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
    valid_mask = np.concatenate([valid_mask, np.ones([sp_boxes.shape[0]],
        dtype=np.bool_)], axis=0)
    boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
    group_ids = np.concatenate([gt_group_ids, sp_group_ids], axis=0)
    if self._enable_global_rot:
        prep.noise_per_object_v3_(boxes, None, valid_mask, 0, 0, self.
            _global_rot_range, group_ids=group_ids, num_try=100)
    sp_boxes_new = boxes[gt_boxes.shape[0]:]
    sp_boxes_bv = box_np_ops.center_to_corner_box2d(sp_boxes_new[:, 0:2],
        sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])
    total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
    coll_mat = prep.box_collision_test(total_bv, total_bv)
    diag = np.arange(total_bv.shape[0])
    coll_mat[diag, diag] = False
    valid_samples = []
    idx = num_gt
    for num in group_num:
        if coll_mat[idx:idx + num].any():
            coll_mat[idx:idx + num] = False
            coll_mat[:, idx:idx + num] = False
        else:
            for i in range(num):
                if self._enable_global_rot:
                    sampled[idx - num_gt + i]['box3d_lidar'][:2] = boxes[
                        idx + i, :2]
                    sampled[idx - num_gt + i]['box3d_lidar'][-1] = boxes[
                        idx + i, -1]
                    sampled[idx - num_gt + i]['rot_transform'] = boxes[idx +
                        i, -1] - sp_boxes[idx + i - num_gt, -1]
                valid_samples.append(sampled[idx - num_gt + i])
        idx += num
    return valid_samples
