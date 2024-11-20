def __init__(self, name, inputs, outputs, weights=None, bias=None, scope=None):
    self.name = name
    self.inputs = inputs
    self.outputs = outputs
    self.weights = weights
    self.bias = bias
    self.scope = scope
