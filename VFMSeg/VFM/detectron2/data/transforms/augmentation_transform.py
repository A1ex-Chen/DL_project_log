def transform(self, tfm: Transform) ->None:
    """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
    self.image = tfm.apply_image(self.image)
    if self.boxes is not None:
        self.boxes = tfm.apply_box(self.boxes)
    if self.sem_seg is not None:
        self.sem_seg = tfm.apply_segmentation(self.sem_seg)
