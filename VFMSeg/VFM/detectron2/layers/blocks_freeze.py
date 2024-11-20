def freeze(self):
    """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
    for p in self.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(self)
    return self
