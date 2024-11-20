def __call__(self, data):
    """ Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
    points = data[None]
    occ = data['occ']
    data_out = data.copy()
    if not self.spatial_completion:
        n_steps, T, dim = points.shape
        N_max = min(self.N, T)
        if self.connected_samples or not self.random:
            indices = np.random.randint(T, size=self.N
                ) if self.random else np.arange(N_max)
            data_out.update({None: points[:, indices], 'occ': occ[:, indices]})
        else:
            indices = np.random.randint(T, size=(n_steps, self.N))
            help_arr = np.arange(n_steps).reshape(-1, 1)
            data_out.update({None: points[help_arr, indices, :], 'occ': occ
                [help_arr, indices]})
    else:
        all_pts = []
        all_occ = []
        for pts, o_value in zip(points, occ):
            N_pts, dim = pts.shape
            indices = np.random.randint(N_pts, size=self.N)
            all_pts.append(pts[indices, :])
            all_occ.append(o_value[indices])
        data_out.update({None: np.stack(all_pts), 'occ': np.stack(all_occ)})
    return data_out
