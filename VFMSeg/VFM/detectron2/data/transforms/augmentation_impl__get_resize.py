def _get_resize(self, image: np.ndarray, scale: float) ->Transform:
    input_size = image.shape[:2]
    target_size = self.target_height, self.target_width
    target_scale_size = np.multiply(target_size, scale)
    output_scale = np.minimum(target_scale_size[0] / input_size[0], 
        target_scale_size[1] / input_size[1])
    output_size = np.round(np.multiply(input_size, output_scale)).astype(int)
    return ResizeTransform(input_size[0], input_size[1], output_size[0],
        output_size[1], self.interp)
