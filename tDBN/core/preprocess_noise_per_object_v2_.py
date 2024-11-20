def noise_per_object_v2_(gt_boxes, points=None, valid_mask=None,
    rotation_perturb=np.pi / 4, center_noise_std=1.0,
    global_random_rot_range=np.pi / 4, num_try=100):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range,
            global_random_rot_range]
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std,
            center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes,
        num_try, 3])
    rot_noises = np.random.uniform(rotation_perturb[0], rotation_perturb[1],
        size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis], size=[num_boxes, num_try])
    origin = [0.5, 0.5, 0]
    gt_box_corners = box_np_ops.center_to_corner_box3d(gt_boxes[:, :3],
        gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2)
    if np.abs(global_random_rot_range[0] - global_random_rot_range[1]) < 0.001:
        selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
            valid_mask, loc_noises, rot_noises)
    else:
        selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
            valid_mask, loc_noises, rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    if points is not None:
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks,
            loc_transforms, rot_transforms, valid_mask)
    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)