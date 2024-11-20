@numba.njit
def corners_3d_jit(dims, origin=0.5):
    ndim = 3
    corners_norm = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
        0, 1, 1, 1, 0, 1, 1, 1], dtype=dims.dtype).reshape((8, 3))
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 **
        ndim, ndim))
    return corners
