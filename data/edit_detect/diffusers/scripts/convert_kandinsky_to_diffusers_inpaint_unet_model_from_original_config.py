def inpaint_unet_model_from_original_config():
    model = UNet2DConditionModel(**INPAINT_UNET_CONFIG)
    return model
