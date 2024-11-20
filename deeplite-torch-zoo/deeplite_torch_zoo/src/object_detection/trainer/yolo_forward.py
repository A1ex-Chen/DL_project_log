def forward(self, x, *args, **kwargs):
    if isinstance(x, dict):
        return self.loss(x, *args, **kwargs)
    return self._forward(x, *args, **kwargs)
