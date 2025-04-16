def get_kitti_image_info(path, training=True, label_info=True, velodyne=
    False, calib=False, image_ids=7481, extend_matrix=True, num_worker=8,
    relative_path=True, with_imageshape=True):
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
        annotations = None
        if velodyne:
            image_info['velodyne_path'] = get_velodyne_path(idx, path,
                training, relative_path)
        image_info['img_path'] = get_image_path(idx, path, training,
            relative_path)
        if with_imageshape:
            img_path = image_info['img_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['img_shape'] = np.array(io.imread(img_path).shape[:2
                ], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        if calib:
            calib_path = get_calib_path(idx, path, training, relative_path=
                False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]
                ).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]
                ).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]
                ).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]
                ).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            image_info['calib/P0'] = P0
            image_info['calib/P1'] = P1
            image_info['calib/P2'] = P2
            image_info['calib/P3'] = P3
            R0_rect = np.array([float(info) for info in lines[4].split(' ')
                [1:10]]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.0
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect
            image_info['calib/R0_rect'] = rect_4x4
            Tr_velo_to_cam = np.array([float(info) for info in lines[5].
                split(' ')[1:13]]).reshape([3, 4])
            Tr_imu_to_velo = np.array([float(info) for info in lines[6].
                split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
            image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
        if annotations is not None:
            image_info['annos'] = annotations
            add_difficulty_to_annos(image_info)
        return image_info
    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)
