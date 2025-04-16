def preprocess(self, image, scale, resample):
    outer, inner = self.size, scale
    factor = self.size // self.mask_size
    if scale > self.size:
        outer, inner = scale, self.size
    top, left = np.random.randint(0, outer - inner + 1), np.random.randint(
        0, outer - inner + 1)
    image = image.resize((scale, scale), resample=resample)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
    mask = np.zeros((self.size // factor, self.size // factor))
    if scale > self.size:
        instance_image = image[top:top + inner, left:left + inner, :]
        mask = np.ones((self.size // factor, self.size // factor))
    else:
        instance_image[top:top + inner, left:left + inner, :] = image
        mask[top // factor + 1:(top + scale) // factor - 1, left // factor +
            1:(left + scale) // factor - 1] = 1.0
    return instance_image, mask
