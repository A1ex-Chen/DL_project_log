def create_kitti_info_file(data_path, save_path=None, create_trainval=False,
    relative_path=True):
    train_img_ids = _read_imageset_file('./kitti/data_split/train.txt')
    val_img_ids = _read_imageset_file('./kitti/data_split/val.txt')
    trainval_img_ids = _read_imageset_file('./kitti/data_split/trainval.txt')
    test_img_ids = _read_imageset_file('./kitti/data_split/test.txt')
    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = pathlib.Path(data_path)
    else:
        save_path = pathlib.Path(save_path)
    kitti_infos_train = kitti.get_kitti_image_info(data_path, training=True,
        velodyne=True, calib=True, image_ids=train_img_ids, relative_path=
        relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / 'kitti_infos_train.pkl'
    print(f'Kitti info train file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = kitti.get_kitti_image_info(data_path, training=True,
        velodyne=True, calib=True, image_ids=val_img_ids, relative_path=
        relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f'Kitti info val file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    """
    if create_trainval:
        kitti_infos_trainval = kitti.get_kitti_image_info(
            data_path,
            training=True,
            velodyne=True,
            calib=True,
            image_ids=trainval_img_ids,
            relative_path=relative_path)
        filename = save_path / 'kitti_infos_trainval.pkl'
        print(f"Kitti info trainval file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_trainval, f)
    """
    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f'Kitti info trainval file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    kitti_infos_test = kitti.get_kitti_image_info(data_path, training=False,
        label_info=False, velodyne=True, calib=True, image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
