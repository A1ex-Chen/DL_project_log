def de_normalize(self, x_in, key):
    return x_in * self.stds[key] + self.means[key]
