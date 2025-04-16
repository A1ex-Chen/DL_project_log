def _get_pad(self, image: np.ndarray) ->Transform:
    input_size = image.shape[:2]
    output_size = self.crop_size
    pad_size = np.subtract(output_size, input_size)
    pad_size = np.maximum(pad_size, 0)
    original_size = np.minimum(input_size, output_size)
    return PadTransform(0, 0, pad_size[1], pad_size[0], original_size[1],
        original_size[0], self.pad_value)
