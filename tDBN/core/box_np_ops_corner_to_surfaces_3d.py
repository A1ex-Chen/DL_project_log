@numba.jit(nopython=False)
def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    surfaces = np.array([[corners[:, 0], corners[:, 1], corners[:, 2],
        corners[:, 3]], [corners[:, 7], corners[:, 6], corners[:, 5],
        corners[:, 4]], [corners[:, 0], corners[:, 3], corners[:, 7],
        corners[:, 4]], [corners[:, 1], corners[:, 5], corners[:, 6],
        corners[:, 2]], [corners[:, 0], corners[:, 4], corners[:, 5],
        corners[:, 1]], [corners[:, 3], corners[:, 2], corners[:, 6],
        corners[:, 7]]]).transpose([2, 0, 1, 3])
    return surfaces
