def build_encoder(config, *args, **kwargs):
    model_name = config['MODEL']['ENCODER']['NAME']
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')
    return model_entrypoints(model_name)(config, *args, **kwargs)
