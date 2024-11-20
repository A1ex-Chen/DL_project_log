def forward(self, *args, **kwargs):
    outputs = self.model(*args, **kwargs)
    return flatten(outputs)
