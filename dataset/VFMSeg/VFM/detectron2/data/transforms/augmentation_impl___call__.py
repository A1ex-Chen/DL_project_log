def __call__(self, aug_input):
    do = self._rand_range() < self.prob
    if do:
        return self.aug(aug_input)
    else:
        return NoOpTransform()
