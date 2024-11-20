def forward(self, x):
    """Normalize predictions of object detection model with input size-dependent factors."""
    xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
    return cls, xywh * self.normalize
