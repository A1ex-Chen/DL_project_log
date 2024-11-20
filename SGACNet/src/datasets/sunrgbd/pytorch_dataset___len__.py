def __len__(self):
    if self.camera is None:
        return len(self.img_dir[self._split]['list'])
    return len(self.img_dir[self._split]['dict'][self.camera])
