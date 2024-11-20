@property
def duration(self):
    return self._samples.shape[0] / float(self._sample_rate)
