def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
    if scale_factor is not None:
        if np.isscalar(scale_factor) and len(input_shape) > 1:
            scale_factor = [scale_factor, scale_factor]
        scale_factor = list(scale_factor)
        scale_factor = [1] * (len(input_shape) - len(scale_factor)
            ) + scale_factor
    if output_shape is not None:
        output_shape = list(input_shape[len(output_shape):]) + list(np.uint
            (np.array(output_shape)))
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(
            scale_factor)))
    return scale_factor, output_shape
