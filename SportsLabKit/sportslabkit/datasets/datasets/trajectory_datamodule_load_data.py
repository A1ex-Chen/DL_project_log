def load_data(self, path):
    data = np.loadtxt(path, delimiter=',')
    return data
