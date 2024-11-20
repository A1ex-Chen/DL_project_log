def _check_download(self, root):
    self.root = Path(root)
    if not self.root.exists():
        self._download()
