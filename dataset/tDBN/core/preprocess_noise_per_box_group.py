@numba.njit
def noise_per_box_group(boxes, valid_mask, loc_noises, rot_noises, group_nums):
    num_groups = group_nums.shape[0]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_corners[i] = box_corners[i + idx]
                    current_corners[i] -= boxes[i + idx, :2]
                    _rotation_box2d_jit_(current_corners[i], rot_noises[idx +
                        i, j], rot_mat_T)
                    current_corners[i] += boxes[i + idx, :2] + loc_noises[i +
                        idx, j, :2]
                coll_mat = box_collision_test(current_corners[:num].reshape
                    (num, 4, 2), box_corners)
                for i in range(num):
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                    break
        idx += num
    return success_mask
