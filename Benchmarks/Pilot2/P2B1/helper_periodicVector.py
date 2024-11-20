def periodicVector(x0, x1, dimensions):
    """
    Calculating periodic distances for the x, y, z dimensions
    """
    for i in range(len(dimensions)):
        delta = x0[:, :, i] - x1[i]
        delta = np.where(delta > 0.5 * dimensions[i], delta - dimensions[i],
            delta)
        delta = np.where(delta < -(0.5 * dimensions[i]), delta + dimensions
            [i], delta)
        x0[:, :, i] = delta * 4
    return x0
