def __init__(self, metric: BaseCostMatrixMetric=IoUCMM(), gate: float=np.inf
    ) ->None:
    if not isinstance(metric, BaseCostMatrixMetric):
        raise TypeError('metric should be a BaseCostMatrixMetric')
    self.metric = metric
    self.gate = gate
