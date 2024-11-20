def __getitem__(self, idx):
    """identity enchange is conducted only in training mode
        """
    if self.mode == 'train':
        id1, id2 = np.random.choice(self.hid, size=2, replace=False)
        m1, m2 = np.random.choice(self.models, size=2)
        data1 = self.get_data_dict(m1, idx, id1, id2)
        data2 = self.get_data_dict(m2, idx, id2, id1)
        return [data1, data2]
    else:
        m = self.models[idx]
        data = self.get_data_dict(m, idx)
        return data
