def ProphetNetLayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedProphetNetLayerNorm
            return FusedProphetNetLayerNorm(normalized_shape, eps,
                elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
