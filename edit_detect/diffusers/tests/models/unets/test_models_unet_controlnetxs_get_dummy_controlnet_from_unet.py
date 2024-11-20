def get_dummy_controlnet_from_unet(self, unet, **kwargs):
    """For some tests we also need the underlying ControlNetXS-Adapter. For these, we'll build the UNetControlNetXSModel from the UNet and ControlNetXS-Adapter"""
    return ControlNetXSAdapter.from_unet(unet, size_ratio=1,
        conditioning_embedding_out_channels=(2, 2), **kwargs)
