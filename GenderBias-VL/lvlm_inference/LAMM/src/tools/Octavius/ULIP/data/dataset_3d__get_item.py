def _get_item(self, index):
    point_set, label = self.list_of_points[index], self.list_of_labels[index]
    point_set = np.concatenate((pc_normalize(point_set[:, :3]), point_set[:,
        3:6] / 255), axis=1)
    if self.use_height:
        self.gravity_dim = 1
        height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1
            ] - point_set[:, self.gravity_dim:self.gravity_dim + 1].min()
        point_set = np.concatenate((point_set, height_array), axis=1)
    return point_set, label[0]
