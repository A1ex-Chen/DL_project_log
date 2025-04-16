def to_onnx_compatible(self):
    self.net.set_swish(memory_efficient=False)
