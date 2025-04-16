def list_openai_models() ->List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_tag_models('openai')
