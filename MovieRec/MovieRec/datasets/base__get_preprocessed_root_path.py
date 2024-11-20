def _get_preprocessed_root_path(self):
    root = self._get_rawdata_root_path()
    return root.joinpath('preprocessed')
