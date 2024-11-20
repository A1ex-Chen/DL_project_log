def load_camera(self, idx, data={}):
    data['camera_mat'] = self.K
    data['scale_mat'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [0, 0, 0, 1]]).astype(np.float32)
    data['idx'] = idx
