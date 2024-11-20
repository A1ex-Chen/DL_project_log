def __len__(self):
    if self.is_training:
        return len(self._captions_reader)
    else:
        return len(self._image_ids)
