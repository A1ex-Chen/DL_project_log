@register_to_config
def __init__(self, do_resize: bool=True, vae_scale_factor: int=8, resample:
    str='lanczos', do_normalize: bool=True, do_binarize: bool=False,
    do_convert_grayscale: bool=False):
    super().__init__(do_resize=do_resize, vae_scale_factor=vae_scale_factor,
        resample=resample, do_normalize=do_normalize, do_binarize=
        do_binarize, do_convert_grayscale=do_convert_grayscale)
