def __init__(self, motion_metric: BaseCostMatrixMetric=IoUCMM(), beta:
    float=0.5, motion_metric_gate: float=np.inf, visual_metric:
    BaseCostMatrixMetric=CosineCMM(), visual_metric_gate: float=np.inf) ->None:
    if not isinstance(motion_metric, BaseCostMatrixMetric):
        raise TypeError('motion_metric should be a BaseCostMatrixMetric')
    if not isinstance(visual_metric, BaseCostMatrixMetric):
        raise TypeError('visual_metric should be a BaseCostMatrixMetric')
    self.motion_metric = motion_metric
    self.motion_metric_gate = motion_metric_gate
    self.visual_metric = visual_metric
    self.visual_metric_gate = visual_metric_gate
    self.beta = beta
