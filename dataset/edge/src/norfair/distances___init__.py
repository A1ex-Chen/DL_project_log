def __init__(self, metric: str='euclidean', **kwargs):
    self.metric = metric
    super().__init__(distance_function=partial(cdist, metric=self.metric,
        **kwargs))
