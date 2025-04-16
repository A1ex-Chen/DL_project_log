@property
def avg(self):
    return np.sum(self.values) / np.sum(self.counts)
