@property
def dummy_cond_unet(self):
    unet = UNet2DModel.from_pretrained('diffusers/consistency-models-test',
        subfolder='test_unet_class_cond')
    return unet
