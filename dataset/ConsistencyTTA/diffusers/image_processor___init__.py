@register_to_config
def __init__(self, do_resize: bool=True, vae_scale_factor: int=8, resample:
    str='lanczos', do_normalize: bool=True):
    super().__init__()
