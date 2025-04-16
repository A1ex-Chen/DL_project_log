def get_model(self, cfg=None, weights=None, verbose=True):
    """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration. Defaults to None.
            weights (str, optional): Path to pre-trained model weights. Defaults to None.
            verbose (bool): Verbose logging if True. Defaults to True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
    model = RTDETRDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and
        RANK == -1)
    if weights:
        model.load(weights)
    return model
