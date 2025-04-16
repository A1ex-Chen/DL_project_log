def get_calib_path(idx, prefix, training=True, relative_path=True,
    exist_check=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
        relative_path, exist_check)
