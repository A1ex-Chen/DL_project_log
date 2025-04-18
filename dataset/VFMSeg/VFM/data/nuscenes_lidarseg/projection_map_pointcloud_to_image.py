def map_pointcloud_to_image(pc, im_shape, info, im=None):
    """
    Maps the lidar point cloud to the image.
    :param pc: (3, N)
    :param im_shape: image to check size and debug
    :param info: dict with calibration infos
    :param im: image, only for visualization
    :return:
    """
    pc = pc.copy()
    pc = Quaternion(info['lidar2ego_rotation']).rotation_matrix @ pc
    pc = pc + np.array(info['lidar2ego_translation'])[:, np.newaxis]
    pc = Quaternion(info['ego2global_rotation_lidar']).rotation_matrix @ pc
    pc = pc + np.array(info['ego2global_translation_lidar'])[:, np.newaxis]
    pc = pc - np.array(info['ego2global_translation_cam'])[:, np.newaxis]
    pc = Quaternion(info['ego2global_rotation_cam']).rotation_matrix.T @ pc
    pc = pc - np.array(info['cam2ego_translation'])[:, np.newaxis]
    pc = Quaternion(info['cam2ego_rotation']).rotation_matrix.T @ pc
    depths = pc[2, :]
    points = view_points(pc, np.array(info['cam_intrinsic']), normalize=True)
    points = points.astype(np.float32)
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < im_shape[1])
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < im_shape[0])
    points = points[:, mask]
    if im is not None:
        coloring = depths
        coloring = coloring[mask]
        plt.figure(figsize=(9, 16))
        plt.imshow(im)
        plt.scatter(points[0, :], points[1, :], c=coloring, s=2)
        plt.axis('off')
    return mask, pc.T, points.T[:, :2]
