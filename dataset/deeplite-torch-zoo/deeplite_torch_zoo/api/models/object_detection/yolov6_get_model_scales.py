def get_model_scales():
    scale_dict = DEFAULT_MODEL_SCALES
    param_names = 'depth_mul', 'width_mul'
    return {cfg_name: dict(zip(param_names, param_cfg)) for cfg_name,
        param_cfg in scale_dict.items()}
