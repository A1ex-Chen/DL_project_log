def __init__(self, dataset, batch_size):
    """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
    self.dataset = dataset
    self.batch_size = batch_size
    self._buckets = [[] for _ in range(2)]
