def get_crop_size(self, image_size):
    """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
    h, w = image_size
    if self.crop_type == 'relative':
        ch, cw = self.crop_size
        return int(h * ch + 0.5), int(w * cw + 0.5)
    elif self.crop_type == 'relative_range':
        crop_size = np.asarray(self.crop_size, dtype=np.float32)
        ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
        return int(h * ch + 0.5), int(w * cw + 0.5)
    elif self.crop_type == 'absolute':
        return min(self.crop_size[0], h), min(self.crop_size[1], w)
    elif self.crop_type == 'absolute_range':
        assert self.crop_size[0] <= self.crop_size[1]
        ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.
            crop_size[1]) + 1)
        cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.
            crop_size[1]) + 1)
        return ch, cw
    else:
        raise NotImplementedError('Unknown crop type {}'.format(self.crop_type)
            )
