def _get_preprocessed_folder_path(self):
    preprocessed_root = self._get_preprocessed_root_path()
    folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}'.format(self.
        code(), self.min_rating, self.min_uc, self.min_sc, self.split)
    return preprocessed_root.joinpath(folder_name)
