def _get_crop(self, image: np.ndarray) ->Transform:
    input_size = image.shape[:2]
    output_size = self.crop_size
    max_offset = np.subtract(input_size, output_size)
    max_offset = np.maximum(max_offset, 0)
    offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
    offset = np.round(offset).astype(int)
    return CropTransform(offset[1], offset[0], output_size[1], output_size[
        0], input_size[1], input_size[0])
