def _check_split(self, split):
    assert split in ['train', 'valid'
        ], f"Split must be in {'train', 'valid'}, got {split}"
    self.split = split
