def __next__(self):
    _random = Random()
    res = {}
    idx = _random.choices(self.population, self.weights)[0]
    res.update(next(self._source_iterators[idx]))
    return res
