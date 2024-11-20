def upsample_bilinear2d(node):
    os = node.outputs[0].shape
    return prod(os) * 4
