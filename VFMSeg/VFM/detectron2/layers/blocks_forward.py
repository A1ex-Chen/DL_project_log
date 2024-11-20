def forward(self, x):
    return self.pointwise(self.depthwise(x))
