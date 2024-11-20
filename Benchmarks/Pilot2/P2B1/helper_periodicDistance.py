def periodicDistance(x0, x1, dimensions):
    com = np.copy(x1)
    com = com.reshape(1, 1, -1)
    com = np.repeat(com, x0.shape[0], axis=0)
    com = np.repeat(com, x0.shape[1], axis=1)
    delta = np.abs(x0 - com)
    delta = np.where(delta > np.multiply(0.5, dimensions), delta -
        dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))
