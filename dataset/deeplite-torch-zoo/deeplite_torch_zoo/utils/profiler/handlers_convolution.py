def convolution(node):
    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
        _, ic, *ks = node.inputs[1].shape
    else:
        ic, _, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    return prod(os) * ic * prod(ks)
