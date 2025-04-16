def __getitem__(self, index):
    points, label = self._get_item(index)
    pt_idxs = np.arange(0, points.shape[0])
    if self.subset == 'train':
        np.random.shuffle(pt_idxs)
    current_points = points[pt_idxs].copy()
    current_points = torch.from_numpy(current_points).float()
    label_name = self.shape_names[int(label)]
    return current_points, label, label_name
