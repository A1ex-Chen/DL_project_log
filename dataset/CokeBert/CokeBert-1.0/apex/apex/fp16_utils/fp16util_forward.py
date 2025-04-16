def forward(self, *inputs):
    inputs = tuple(t.half() for t in inputs)
    return self.network(*inputs)
