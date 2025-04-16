def compute_mask_weights(self, region: DiffusionRegion) ->torch.tensor:
    """Computes a tensor of weights for a given diffusion region"""
    MASK_BUILDERS = {MaskModes.CONSTANT.value: self._constant_weights,
        MaskModes.GAUSSIAN.value: self._gaussian_weights, MaskModes.QUARTIC
        .value: self._quartic_weights}
    return MASK_BUILDERS[region.mask_type](region)
