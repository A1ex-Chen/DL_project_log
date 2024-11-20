@lru_cache(maxsize=4)
def _get_grid(size, w, h, polar=False):
    """
    Construct the grid of points.

    Points are choosen
    Results are cached since the grid in absolute coordinates doesn't change.
    """
    step = np.pi / size
    start = -np.pi / 2 + step / 2
    end = np.pi / 2
    theta, fi = np.mgrid[start:end:step, start:end:step]
    if polar:
        tan_theta = np.tan(theta)
        X = tan_theta * np.cos(fi)
        Y = tan_theta * np.sin(fi)
    else:
        X = np.tan(fi)
        Y = np.divide(np.tan(theta), np.cos(fi))
    points = np.vstack((X.flatten(), Y.flatten())).T
    return points * max(h, w) + np.array([w // 2, h // 2])
