def get_model_scales(_model_key):
    scale_dict = CUSTOM_MODEL_SCALES.get(_model_key, DEFAULT_MODEL_SCALES)
    param_names = 'depth_mul', 'width_mul', 'max_channels'
    return {cfg_name: dict(zip(param_names, param_cfg)) for cfg_name,
        param_cfg in scale_dict.items()}
