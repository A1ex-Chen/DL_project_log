def sparse_to_dense(voxel_data, dims, dtype=np.bool):
    if voxel_data.ndim != 2 or voxel_data.shape[0] != 3:
        raise ValueError('voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims] * 3
    dims = np.atleast_2d(dims).T
    xyz = voxel_data.astype(np.int)
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:, valid_ix]
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out
