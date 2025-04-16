def get_pretrained_url(model: str, tag: str):
    if model not in _PRETRAINED:
        return ''
    model_pretrained = _PRETRAINED[model]
    if tag not in model_pretrained:
        return ''
    return model_pretrained[tag]
