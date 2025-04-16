def __next__(self) ->NDArray[np.uint8]:
    if self._index < len(self):
        img = self.imgs[self._index]
        self._index += 1
        return img
    raise StopIteration
