def __init__(self, detection_model, motion_model, matching_fn:
    SimpleMatchingFunction=SimpleMatchingFunction(metric=IoUCMM(
    use_pred_box=True), gate=1.0), window_size: int=1, step_size: (int |
    None)=None, max_staleness: int=5, min_length: int=5, callbacks=None):
    """
        Initializes the SORT Tracker.

        Args:
            detection_model (Any): The model used for object detection.
            motion_model (Any): The model used for motion prediction.
            metric (IoUCMM, optional): The metric used for matching. Defaults to IoUCMM().
            metric_gate (float, optional): The gating threshold for the metric. Defaults to 1.0.
        """
    super().__init__(window_size=window_size, step_size=step_size,
        max_staleness=max_staleness, min_length=min_length, callbacks=callbacks
        )
    self.detection_model = detection_model
    self.motion_model = motion_model
    self.matching_fn = matching_fn
