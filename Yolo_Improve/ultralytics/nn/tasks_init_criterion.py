def init_criterion(self):
    """Initialize the loss criterion for the RTDETRDetectionModel."""
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)
