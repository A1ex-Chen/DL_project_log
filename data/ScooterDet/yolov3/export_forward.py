def forward(self, x):
    xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
    return cls * conf, xywh * self.normalize
