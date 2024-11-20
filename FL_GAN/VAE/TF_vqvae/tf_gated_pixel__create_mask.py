def _create_mask(num_inputs, num_outputs, kernel_shape, data_num_channels,
    mask_type):
    """
    Produces a causal mask of the given type and shape
    """
    mask_type = mask_type.lower()
    kernel_h, kernel_w = kernel_shape
    center_h = kernel_h // 2
    center_w = kernel_w // 2
    mask = np.ones((kernel_h, kernel_w, num_inputs, num_outputs), dtype=np.
        float32)
    if mask_type == 'v':
        mask[center_h:, :, :, :] = 0.0
    else:
        mask[center_h, center_w + 1:, :, :] = 0.0
        mask[center_h + 1:, :, :, :] = 0.0
        if mask_type == 'b':
            mask_pixel = lambda i, j: i > j
        else:
            mask_pixel = lambda i, j: i >= j
        for i in range(num_inputs):
            for j in range(num_outputs):
                if mask_pixel(i % data_num_channels, j % data_num_channels):
                    mask[center_h, center_w, i, j] = 0.0
    return mask
