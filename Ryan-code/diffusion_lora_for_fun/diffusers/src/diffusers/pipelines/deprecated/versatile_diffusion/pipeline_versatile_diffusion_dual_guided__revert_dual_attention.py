def _revert_dual_attention(self):
    """
        Revert the image_unet `DualTransformer2DModel` blocks back to `Transformer2DModel` with image_unet weights Call
        this function if you reuse `image_unet` in another pipeline, e.g. `VersatileDiffusionPipeline`
        """
    for name, module in self.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            parent_name, index = name.rsplit('.', 1)
            index = int(index)
            self.image_unet.get_submodule(parent_name)[index
                ] = module.transformers[0]
    self.image_unet.register_to_config(dual_cross_attention=False)
