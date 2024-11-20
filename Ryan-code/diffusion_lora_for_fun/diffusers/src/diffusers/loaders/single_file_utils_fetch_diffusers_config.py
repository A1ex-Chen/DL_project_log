def fetch_diffusers_config(checkpoint):
    model_type = infer_diffusers_model_type(checkpoint)
    model_path = DIFFUSERS_DEFAULT_PIPELINE_PATHS[model_type]
    return model_path
