def super_res_unet_first_steps_model_from_original_config():
    model = UNet2DModel(**SUPER_RES_UNET_FIRST_STEPS_CONFIG)
    return model
