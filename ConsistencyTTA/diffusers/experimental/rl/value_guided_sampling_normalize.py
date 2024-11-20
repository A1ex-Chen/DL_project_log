def normalize(self, x_in, key):
    return (x_in - self.means[key]) / self.stds[key]
