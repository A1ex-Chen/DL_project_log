def leaky_relu(node):
    os = node.outputs[0].shape
    return prod(os)
