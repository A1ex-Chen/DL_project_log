def __init__(self, detection_model=None, image_model=None, motion_model=
    None, matching_fn: MotionVisualMatchingFunction=
    MotionVisualMatchingFunction(motion_metric=IoUCMM(), motion_metric_gate
    =0.2, visual_metric=CosineCMM(), visual_metric_gate=0.2, beta=0.5),
    window_size: int=1, step_size: (int | None)=None, max_staleness: int=5,
    min_length: int=5, callbacks=None):
    super().__init__(window_size=window_size, step_size=step_size,
        max_staleness=max_staleness, min_length=min_length, callbacks=callbacks
        )
    self.detection_model = detection_model
    self.image_model = image_model
    self.motion_model = motion_model
    self.matching_fn = matching_fn
