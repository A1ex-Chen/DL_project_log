def getstate(self):
    return {'input_states': tuple(iterator.getstate() for iterator in self.
        _source_iterators)}
