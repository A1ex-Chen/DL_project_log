def __repr__(self):
    text = 'Node (name: {}, inputs: {}, outputs: {}, w: {}, b: {})'.format(self
        .name, len(self.inputs), len(self.outputs), self.weights, self.bias)
    return text
