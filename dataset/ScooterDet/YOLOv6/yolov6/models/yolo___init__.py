def __init__(self, config, channels=3, num_classes=None, fuse_ab=False,
    distill_ns=False):
    super().__init__()
    num_layers = config.model.head.num_layers
    self.backbone, self.neck, self.detect = build_network(config, channels,
        num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)
    self.stride = self.detect.stride
    self.detect.initialize_biases()
    initialize_weights(self)
