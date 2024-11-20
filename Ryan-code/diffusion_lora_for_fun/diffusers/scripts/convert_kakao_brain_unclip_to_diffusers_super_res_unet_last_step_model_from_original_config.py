def super_res_unet_last_step_model_from_original_config():
    model = UNet2DModel(**SUPER_RES_UNET_LAST_STEP_CONFIG)
    return model
