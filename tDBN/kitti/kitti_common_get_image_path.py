def get_image_path(idx, prefix, training=True, relative_path=True,
    exist_check=True):
    return get_kitti_info_path(idx, prefix, 'image_2', '.png', training,
        relative_path, exist_check)
