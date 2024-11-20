def __getitem__(self, index):
    data_dict = self.data[index].copy()
    lidar_front_center = np.load(data_dict['lidar_path'])
    points = lidar_front_center['points']
    if 'row' not in lidar_front_center.keys():
        print('row not in lidar dict, return None, {}'.format(data_dict[
            'lidar_path']))
        return {}
    rows = lidar_front_center['row'].astype(np.int)
    cols = lidar_front_center['col'].astype(np.int)
    label_img = np.array(Image.open(data_dict['label_path']))
    label_img = undistort_image(self.config, label_img, 'front_center')
    label_pc = label_img[rows, cols, :]
    seg_label = np.full(label_pc.shape[0], fill_value=len(self.
        rgb_to_cls_idx), dtype=np.int64)
    for rgb_values, cls_idx in self.rgb_to_cls_idx.items():
        idx = (rgb_values == label_pc).all(1)
        if idx.any():
            seg_label[idx] = cls_idx
    image = Image.open(data_dict['camera_path'])
    image_size = image.size
    assert image_size == (1920, 1208)
    image = undistort_image(self.config, np.array(image), 'front_center')
    points_img = np.stack([lidar_front_center['row'], lidar_front_center[
        'col']], 1).astype(np.float32)
    assert np.all(points_img[:, 0] < image_size[1])
    assert np.all(points_img[:, 1] < image_size[0])
    data_dict['seg_label'] = seg_label.astype(np.uint8)
    data_dict['points'] = points.astype(np.float32)
    data_dict['points_img'] = points_img
    data_dict['img'] = image
    return data_dict
