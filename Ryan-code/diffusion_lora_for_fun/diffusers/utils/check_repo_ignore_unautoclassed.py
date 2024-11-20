def ignore_unautoclassed(model_name):
    """Rules to determine if `name` should be in an auto class."""
    if model_name in IGNORE_NON_AUTO_CONFIGURED:
        return True
    if 'Encoder' in model_name or 'Decoder' in model_name:
        return True
    return False
