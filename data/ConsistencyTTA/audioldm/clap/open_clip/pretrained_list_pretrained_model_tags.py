def list_pretrained_model_tags(model: str):
    """return all pretrain tags for the specified model architecture"""
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags
