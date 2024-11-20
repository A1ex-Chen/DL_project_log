def random_crop_frustum(bboxes, rect, Trv2c, P2, max_crop_height=1.0,
    max_crop_width=0.9):
    num_gt = bboxes.shape[0]
    crop_minxy = np.random.uniform([1 - max_crop_width, 1 - max_crop_height
        ], [0.3, 0.3], size=[num_gt, 2])
    crop_maxxy = np.ones([num_gt, 2], dtype=bboxes.dtype)
    crop_bboxes = np.concatenate([crop_minxy, crop_maxxy], axis=1)
    left = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if left:
        crop_bboxes[:, [0, 2]] -= crop_bboxes[:, 0:1]
    crop_bboxes *= np.tile(bboxes[:, 2:] - bboxes[:, :2], [1, 2])
    crop_bboxes += np.tile(bboxes[:, :2], [1, 2])
    C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
    frustums = box_np_ops.get_frustum_v2(crop_bboxes, C)
    frustums -= T
    frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
    frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
    return frustums
