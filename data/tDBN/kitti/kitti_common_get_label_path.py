def get_label_path(idx, prefix, training=True, relative_path=True,
    exist_check=True):
    return get_kitti_info_path(idx, prefix, 'label_2', '.txt', training,
        relative_path, exist_check)
