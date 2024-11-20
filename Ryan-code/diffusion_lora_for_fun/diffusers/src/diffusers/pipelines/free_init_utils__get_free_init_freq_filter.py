def _get_free_init_freq_filter(self, shape: Tuple[int, ...], device: Union[
    str, torch.dtype], filter_type: str, order: float,
    spatial_stop_frequency: float, temporal_stop_frequency: float
    ) ->torch.Tensor:
    """Returns the FreeInit filter based on filter type and other input conditions."""
    time, height, width = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if spatial_stop_frequency == 0 or temporal_stop_frequency == 0:
        return mask
    if filter_type == 'butterworth':

        def retrieve_mask(x):
            return 1 / (1 + (x / spatial_stop_frequency ** 2) ** order)
    elif filter_type == 'gaussian':

        def retrieve_mask(x):
            return math.exp(-1 / (2 * spatial_stop_frequency ** 2) * x)
    elif filter_type == 'ideal':

        def retrieve_mask(x):
            return 1 if x <= spatial_stop_frequency * 2 else 0
    else:
        raise NotImplementedError(
            '`filter_type` must be one of gaussian, butterworth or ideal')
    for t in range(time):
        for h in range(height):
            for w in range(width):
                d_square = (spatial_stop_frequency /
                    temporal_stop_frequency * (2 * t / time - 1)) ** 2 + (2 *
                    h / height - 1) ** 2 + (2 * w / width - 1) ** 2
                mask[..., t, h, w] = retrieve_mask(d_square)
    return mask.to(device)
