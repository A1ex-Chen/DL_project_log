def build_backbone(config, **kwargs):
    model_name = config['MODEL']['BACKBONE']['NAME']
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')
    return model_entrypoints(model_name)(config, **kwargs)
