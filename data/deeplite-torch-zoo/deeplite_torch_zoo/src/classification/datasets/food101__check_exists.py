def _check_exists(self) ->bool:
    return all(folder.exists() and folder.is_dir() for folder in (self.
        _meta_folder, self._images_folder))
