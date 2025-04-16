@register_model
def beit3_large_patch16_224(pretrained=False, **kwargs):
    args = _get_large_config(img_size=224, **kwargs)
    model = BEiT3Wrapper(args, **kwargs)
    return model
