@numba.njit
def noise_per_box_group_v2_(boxes, valid_mask, loc_noises, rot_noises,
    group_nums, global_rot_noises):
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((max_group_num, 2), dtype=boxes.dtype)
    current_grot = np.zeros((max_group_num,), dtype=boxes.dtype)
    dst_grot = np.zeros((max_group_num,), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_box[0, :] = boxes[i + idx]
                    current_radius = np.sqrt(current_box[0, 0] ** 2 + 
                        current_box[0, 1] ** 2)
                    current_grot[i] = np.arctan2(current_box[0, 0],
                        current_box[0, 1])
                    dst_grot[i] = current_grot[i] + global_rot_noises[idx +
                        i, j]
                    dst_pos[i, 0] = current_radius * np.sin(dst_grot[i])
                    dst_pos[i, 1] = current_radius * np.cos(dst_grot[i])
                    current_box[0, :2] = dst_pos[i]
                    current_box[0, -1] += dst_grot[i] - current_grot[i]
                    rot_sin = np.sin(current_box[0, -1])
                    rot_cos = np.cos(current_box[0, -1])
                    rot_mat_T[0, 0] = rot_cos
                    rot_mat_T[0, 1] = -rot_sin
                    rot_mat_T[1, 0] = rot_sin
                    rot_mat_T[1, 1] = rot_cos
                    current_corners[i] = current_box[0, 2:4
                        ] * corners_norm @ rot_mat_T + current_box[0, :2]
                    current_corners[i] -= current_box[0, :2]
                    _rotation_box2d_jit_(current_corners[i], rot_noises[idx +
                        i, j], rot_mat_T)
                    current_corners[i] += current_box[0, :2] + loc_noises[i +
                        idx, j, :2]
                coll_mat = box_collision_test(current_corners[:num].reshape
                    (num, 4, 2), box_corners)
                for i in range(num):
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                        loc_noises[i + idx, j, :2] += dst_pos[i] - boxes[i +
                            idx, :2]
                        rot_noises[i + idx, j] += dst_grot[i] - current_grot[i]
                    break
        idx += num
    return success_mask
