def fused_layer_norm_affine(input, normalized_shape, weight, bias, eps=1e-06):
    return FusedLayerNormAffineFunction.apply(input, weight, bias,
        normalized_shape, eps)
