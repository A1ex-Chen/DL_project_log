def _download(self) ->None:
    if self._check_exists():
        return
    download_and_extract_archive(self.url, download_root=self.root)
