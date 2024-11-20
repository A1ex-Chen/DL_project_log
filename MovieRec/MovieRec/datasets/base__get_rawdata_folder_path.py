def _get_rawdata_folder_path(self):
    root = self._get_rawdata_root_path()
    return root.joinpath(self.raw_code())
