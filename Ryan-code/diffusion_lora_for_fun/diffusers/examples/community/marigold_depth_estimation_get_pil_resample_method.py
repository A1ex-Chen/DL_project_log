def get_pil_resample_method(method_str: str) ->Resampling:
    resample_method_dic = {'bilinear': Resampling.BILINEAR, 'bicubic':
        Resampling.BICUBIC, 'nearest': Resampling.NEAREST}
    resample_method = resample_method_dic.get(method_str, None)
    if resample_method is None:
        raise ValueError(f'Unknown resampling method: {resample_method}')
    else:
        return resample_method
