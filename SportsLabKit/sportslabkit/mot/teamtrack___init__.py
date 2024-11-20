def __init__(self, detection_model=None, image_model=None, motion_model=
    None, calibration_model=None, first_matching_fn:
    MotionVisualMatchingFunction=MotionVisualMatchingFunction(motion_metric
    =IoUCMM(use_pred_box=True), motion_metric_gate=0.2, visual_metric=
    CosineCMM(), visual_metric_gate=0.2, beta=0.5), second_matching_fn=
    SimpleMatchingFunction(metric=IoUCMM(use_pred_box=True), gate=0.9),
    detection_score_threshold=0.6, window_size: int=1, step_size: (int |
    None)=None, max_staleness: int=5, min_length: int=5, callbacks=None):
    super().__init__(window_size=window_size, step_size=step_size,
        max_staleness=max_staleness, min_length=min_length, callbacks=callbacks
        )
    self.detection_model = detection_model
    self.image_model = image_model
    self.motion_model = motion_model
    self.calibration_model = calibration_model
    self.first_matching_fn = first_matching_fn
    self.second_matching_fn = second_matching_fn
    self.detection_score_threshold = detection_score_threshold
    self.homographies = []
