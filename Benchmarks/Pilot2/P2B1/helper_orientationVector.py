def orientationVector(x0, x1, dimensions):
    """
    Calculating the orientation vector for a molecule
    """
    x = np.copy(x0)
    for i in range(len(dimensions)):
        delta = x0[:, :, i] - x1[i]
        delta = np.where(delta > 0.5 * dimensions[i], delta - dimensions[i],
            delta)
        delta = np.where(delta < -(0.5 * dimensions[i]), delta + dimensions
            [i], delta)
        x[:, :, i] = delta
    return x
