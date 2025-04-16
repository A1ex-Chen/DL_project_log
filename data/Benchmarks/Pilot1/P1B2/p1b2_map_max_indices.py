def map_max_indices(nparray):

    def maxi(a):
        return a.argmax()
    return np.array([maxi(a) for a in nparray])
