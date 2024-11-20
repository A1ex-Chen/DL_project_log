def test_StableDiffusionMixin_component(self):
    """Any pipeline that have LDMFuncMixin should have vae and unet components."""
    if not issubclass(self.pipeline_class, StableDiffusionMixin):
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    self.assertTrue(hasattr(pipe, 'vae') and isinstance(pipe.vae, (
        AutoencoderKL, AutoencoderTiny)))
    self.assertTrue(hasattr(pipe, 'unet') and isinstance(pipe.unet, (
        UNet2DConditionModel, UNet3DConditionModel, I2VGenXLUNet,
        UNetMotionModel, UNetControlNetXSModel)))
