def wibbledmean(array):
    return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)
