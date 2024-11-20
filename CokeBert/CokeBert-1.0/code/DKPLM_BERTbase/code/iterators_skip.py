def skip(self, num_to_skip):
    """Fast-forward the iterator by skipping *num_to_skip* elements."""
    next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
    return self
