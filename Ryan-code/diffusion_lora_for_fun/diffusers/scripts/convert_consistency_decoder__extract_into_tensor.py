def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr[timesteps].float()
    dims_to_append = len(broadcast_shape) - len(res.shape)
    return res[(...,) + (None,) * dims_to_append]
