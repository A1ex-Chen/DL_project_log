def _create_reduced_point_cloud(data_path, info_path, save_path=None, back=
    False):
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos):
        v_path = info['velodyne_path']
        v_path = pathlib.Path(data_path) / v_path
        points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1
            ).reshape([-1, 4])
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c,
            P2, info['img_shape'])
        if save_path is None:
            save_filename = v_path.parent.parent / (v_path.parent.stem +
                '_reduced') / v_path.name
            if back:
                save_filename += '_back'
        else:
            save_filename = str(pathlib.Path(save_path) / v_path.name)
            if back:
                save_filename += '_back'
        with open(save_filename, 'w') as f:
            points_v.tofile(f)
