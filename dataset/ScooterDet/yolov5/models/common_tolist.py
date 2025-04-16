def tolist(self):
    r = range(self.n)
    x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.
        times, self.names, self.s) for i in r]
    return x
