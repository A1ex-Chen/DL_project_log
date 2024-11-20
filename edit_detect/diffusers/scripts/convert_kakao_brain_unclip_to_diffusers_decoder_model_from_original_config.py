def decoder_model_from_original_config():
    model = UNet2DConditionModel(**DECODER_CONFIG)
    return model
