def activation(self):
    return {'silu': lambda : nn.SiLU(inplace=True), 'relu': lambda : nn.
        ReLU(inplace=True), 'onnx-silu': ONNXSiLU}[self.config.activation]()
