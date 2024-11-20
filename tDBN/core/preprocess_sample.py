def sample(self, num):
    indices = self._sample(num)
    return [self._sampled_list[i] for i in indices]
