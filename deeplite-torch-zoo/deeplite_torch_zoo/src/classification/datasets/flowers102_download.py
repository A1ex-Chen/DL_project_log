def download(self):
    if self._check_integrity():
        return
    download_and_extract_archive(
        f"{self._download_url_prefix}{self._file_dict['image'][0]}", str(
        self._base_folder), md5=self._file_dict['image'][1])
    for id in ['label', 'setid']:
        filename, md5 = self._file_dict[id]
        download_url(self._download_url_prefix + filename, str(self.
            _base_folder), md5=md5)
