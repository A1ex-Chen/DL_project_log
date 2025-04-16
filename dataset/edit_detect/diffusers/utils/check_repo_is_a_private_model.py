def is_a_private_model(model):
    """Returns True if the model should not be in the main init."""
    if model in PRIVATE_MODELS:
        return True
    if model.endswith('Wrapper'):
        return True
    if model.endswith('Encoder'):
        return True
    if model.endswith('Decoder'):
        return True
    return False
