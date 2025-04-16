def minmax_to_center_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center_min = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center_min
    center = center_min + 0.5 * dims
    return np.concatenate([center, dims], axis=-1)
