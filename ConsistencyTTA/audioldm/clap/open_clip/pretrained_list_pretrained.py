def list_pretrained(as_str: bool=False):
    """returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [(':'.join([k, t]) if as_str else (k, t)) for k in _PRETRAINED.
        keys() for t in _PRETRAINED[k].keys()]
