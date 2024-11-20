def unet_model_from_original_config():
    model = UNet2DConditionModel(**UNET_CONFIG)
    return model
