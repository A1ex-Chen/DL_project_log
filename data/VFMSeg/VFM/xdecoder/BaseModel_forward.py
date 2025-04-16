def forward(self, *inputs, **kwargs):
    outputs = self.model(*inputs, **kwargs)
    return outputs
