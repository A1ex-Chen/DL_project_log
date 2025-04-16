def merge(self, other):
    if isinstance(other, Detections):
        other = other.preds
    if len(other) == 0:
        return self
    pred = np.concatenate((self.preds, other), axis=0)
    return Detections(pred, self.im, self.names, self.times)
