def get_models():
    """Returns the mapping from model type name to Keras model."""
    return {'efficientnet': efficientnet_model.EfficientNet.from_name}
