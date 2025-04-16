def __getitem__(self, idx):
    return _read_and_prep_v9(info=self._kitti_infos[idx], root_path=self.
        _root_path, num_point_features=self._num_point_features, prep_func=
        self._prep_func)
