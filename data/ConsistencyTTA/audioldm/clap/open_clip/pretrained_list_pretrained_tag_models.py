def list_pretrained_tag_models(tag: str):
    """return all models having the specified pretrain tag"""
    models = []
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models
