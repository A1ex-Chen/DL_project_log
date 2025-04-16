def __init__(self, src: Union[torch.Tensor, str, os.PathLike, pathlib.
    PurePath, List[Union[torch.Tensor, Image.Image, str, os.PathLike,
    pathlib.PurePath]]], size: Union[int, Tuple[int, int], list[int]],
    repeat: int=1, repeat_type: str=REPEAT_BY_INPLACE, noise_map_type: str=
    NOISE_MAP_SAME, vmin_out: float=-1.0, vmax_out: float=1.0, generator:
    Union[int, torch.Generator]=0, image_exts: list[str]=['png', 'jpg',
    'jpeg', 'webp'], ext_case_sensitive: bool=False, njob: int=-1):
    if isinstance(src, list):
        self.__src__ = src
    elif isinstance(src, torch.Tensor):
        self.__src__ = torch.split(src, 1)
    elif isinstance(src, str) or isinstance(src, os.PathLike) or isinstance(src
        , pathlib.PurePath):
        self.__src__ = list_all_images(root=src, image_exts=image_exts,
            ext_case_sensitive=ext_case_sensitive)
    else:
        raise TypeError(
            f'Arguement src should not be {type(src)}, it should be Union[torch.Tensor, str, os.PathLike, pathlib.PurePath, List[Union[torch.Tensor, Image.Image, str, os.PathLike, pathlib.PurePath]]]'
            )
    self.__repeat__: int = repeat
    self.__repeat_type__: str = repeat_type
    self.__noise_map_type__: str = noise_map_type
    self.__generator__ = set_generator(generator=generator)
    self.__vmin_out__: float = vmin_out
    self.__vmax_out__: float = vmax_out
    self.__size__: Union[int, Tuple[int], List[int]] = size
    self.__njob__: int = njob
    self.__unique_src_n__: int = len(self)
    self.__build_src_noise__()
