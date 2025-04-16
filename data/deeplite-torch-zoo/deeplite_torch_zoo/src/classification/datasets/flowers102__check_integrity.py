def _check_integrity(self):
    if not (self._images_folder.exists() and self._images_folder.is_dir()):
        return False
    for id in ['label', 'setid']:
        filename, md5 = self._file_dict[id]
        if not check_integrity(str(self._base_folder / filename), md5):
            return False
    return True
