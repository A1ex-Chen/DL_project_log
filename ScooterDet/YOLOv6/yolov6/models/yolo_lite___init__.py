def __init__(self, config, channels=3, num_classes=None):
    super().__init__()
    self.backbone, self.neck, self.detect = build_network(config, channels,
        num_classes)
    self.stride = self.detect.stride
    self.detect.initialize_biases()
    initialize_weights(self)
