def __init__(self, data, loc=(0.0, 0.0, 0.0), scale=1):
    assert data.shape[0] == data.shape[1] == data.shape[2]
    data = np.asarray(data, dtype=np.bool)
    loc = np.asarray(loc)
    self.data = data
    self.loc = loc
    self.scale = scale
