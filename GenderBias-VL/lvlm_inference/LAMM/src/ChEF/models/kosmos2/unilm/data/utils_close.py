def close(self):
    for it in self._source_iterators:
        it.close()
