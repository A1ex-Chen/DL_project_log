def _get_preprocessed_dataset_path(self):
    folder = self._get_preprocessed_folder_path()
    return folder.joinpath('dataset.pkl')
