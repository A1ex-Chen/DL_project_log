def norm(node):
    if node.operator in ['aten::batch_norm', 'aten::instance_norm']:
        affine = node.inputs[1].shape is not None
    elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
        affine = node.inputs[2].shape is not None
    else:
        raise ValueError(node.operator)
    os = node.outputs[0].shape
    return prod(os) if affine else 0
