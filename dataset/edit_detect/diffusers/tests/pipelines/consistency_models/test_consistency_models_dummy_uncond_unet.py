@property
def dummy_uncond_unet(self):
    unet = UNet2DModel.from_pretrained('diffusers/consistency-models-test',
        subfolder='test_unet')
    return unet
