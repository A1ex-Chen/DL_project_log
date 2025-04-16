def fused_layer_norm(input, normalized_shape, eps=1e-06):
    return FusedLayerNormFunction.apply(input, normalized_shape, eps)
