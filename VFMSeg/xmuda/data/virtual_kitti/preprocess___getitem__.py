def __getitem__(self, index):
    data_dict = self.data[index].copy()
    point_cloud = np.load(data_dict['lidar_path'])
    points = point_cloud[:, :3].astype(np.float32)
    labels = point_cloud[:, 6].astype(np.uint8)
    data_dict['seg_label'] = labels
    data_dict['points'] = points
    return data_dict
