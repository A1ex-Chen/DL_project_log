def accumulate(self):
    if len(self.framerate) > 1:
        return np.average(self.framerate)
    else:
        return 0.0
