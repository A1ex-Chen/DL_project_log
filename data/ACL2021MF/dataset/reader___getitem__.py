def __getitem__(self, image_id: int):
    i = self._image_ids[image_id]
    if image_id in self.cache:
        return self.cache[image_id]
    with h5py.File(self._boxes_h5path, 'r') as boxes_h5:
        self.process_single_image(image_id, i, boxes_h5)
    d = self.cache[image_id]
    return {key: np.array(d[key], copy=True) for key in d}
