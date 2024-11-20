def prior_image_model_from_original_config():
    model = PriorTransformer(**PRIOR_IMAGE_CONFIG)
    return model
