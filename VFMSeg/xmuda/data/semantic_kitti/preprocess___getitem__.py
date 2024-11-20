def __getitem__(self, index):
    data_dict = self.data[index].copy()
    scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :3]
    if osp.exists(data_dict['label_path']):
        label = np.fromfile(data_dict['label_path'], dtype=np.uint32)
        label = label.reshape(-1)
        label = label & 65535
    else:
        label = None
    image = Image.open(data_dict['camera_path'])
    image_size = image.size
    keep_idx = points[:, 0] > 0
    points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.
        sum(), 1], dtype=np.float32)], axis=1)
    img_points = (data_dict['proj_matrix'] @ points_hcoords.T).T
    img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)
    keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *
        image_size)
    keep_idx[keep_idx] = keep_idx_img_pts
    img_points = np.fliplr(img_points)
    if label is not None:
        data_dict['seg_label'] = label[keep_idx].astype(np.int16)
    data_dict['points'] = points[keep_idx]
    data_dict['points_img'] = img_points[keep_idx_img_pts]
    data_dict['image_size'] = np.array(image_size)
    return data_dict
