def __init__(self, source_iterators, weights):
    """
        Args:
                source_iterators: list of iterators to zip, item by item
        """
    for source_iterator in source_iterators:
        if not isinstance(source_iterator, iterators.CheckpointableIterator):
            raise ValueError(
                'all iterators in source_iterators have to be CheckpointableIterator'
                )
    self._source_iterators = source_iterators
    assert len(weights) == len(source_iterators)
    self.weights = weights
    self.population = list(range(len(source_iterators)))
