def _download(self) ->None:
    if self._check_exists():
        return
    download_and_extract_archive(self._URL, download_root=self.root, md5=
        self._MD5)
