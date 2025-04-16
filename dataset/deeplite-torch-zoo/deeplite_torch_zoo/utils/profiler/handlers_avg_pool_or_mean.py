def avg_pool_or_mean(node):
    os = node.outputs[0].shape
    return prod(os)
