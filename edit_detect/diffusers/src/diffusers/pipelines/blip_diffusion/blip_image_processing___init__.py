def __init__(self, do_resize: bool=True, size: Dict[str, int]=None,
    resample: PILImageResampling=PILImageResampling.BICUBIC, do_rescale:
    bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize:
    bool=True, image_mean: Optional[Union[float, List[float]]]=None,
    image_std: Optional[Union[float, List[float]]]=None, do_convert_rgb:
    bool=True, do_center_crop: bool=True, **kwargs) ->None:
    super().__init__(**kwargs)
    size = size if size is not None else {'height': 224, 'width': 224}
    size = get_size_dict(size, default_to_square=True)
    self.do_resize = do_resize
    self.size = size
    self.resample = resample
    self.do_rescale = do_rescale
    self.rescale_factor = rescale_factor
    self.do_normalize = do_normalize
    self.image_mean = (image_mean if image_mean is not None else
        OPENAI_CLIP_MEAN)
    self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
    self.do_convert_rgb = do_convert_rgb
    self.do_center_crop = do_center_crop
