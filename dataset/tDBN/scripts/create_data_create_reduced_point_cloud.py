def create_reduced_point_cloud(data_path, train_info_path=None,
    val_info_path=None, test_info_path=None, save_path=None, with_back=False):
    if train_info_path is None:
        train_info_path = pathlib.Path(data_path) / 'kitti_infos_train.pkl'
    if val_info_path is None:
        val_info_path = pathlib.Path(data_path) / 'kitti_infos_val.pkl'
    if test_info_path is None:
        test_info_path = pathlib.Path(data_path) / 'kitti_infos_test.pkl'
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(data_path, train_info_path, save_path,
            back=True)
        _create_reduced_point_cloud(data_path, val_info_path, save_path,
            back=True)
        _create_reduced_point_cloud(data_path, test_info_path, save_path,
            back=True)
