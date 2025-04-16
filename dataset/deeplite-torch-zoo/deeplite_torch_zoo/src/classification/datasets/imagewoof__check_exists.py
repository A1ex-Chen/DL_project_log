def _check_exists(self) ->bool:
    return all(folder.exists() and folder.is_dir() for folder in (self.
        _base_folder,))
