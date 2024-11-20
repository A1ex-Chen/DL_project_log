def _calculate_num_points_in_gt(data_path, infos, relative_path,
    remove_outside=True, num_features=4):
    for info in infos:
        if relative_path:
            v_path = str(pathlib.Path(data_path) / info['velodyne_path'])
        else:
            v_path = info['velodyne_path']
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([
            -1, num_features])
        rect = info['calib/R0_rect']
        Trv2c = info['calib/Tr_velo_to_cam']
        P2 = info['calib/P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(points_v, rect,
                Trv2c, P2, info['img_shape'])
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
            axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera,
            rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([
            num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)
