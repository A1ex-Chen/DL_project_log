def _select_transform(transform, indices):
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=
        transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result
