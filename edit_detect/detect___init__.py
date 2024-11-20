def __init__(self, root: Union[str, os.PathLike, pathlib.PurePath], size:
    Union[int, Tuple[int, int], List[int]], dir_label_map: Dict[str, int]={
    'real': 0, 'fake': 1}, vmin: float=-1.0, vmax: float=1.0, generator:
    Union[int, torch.Generator]=0, image_exts: List[str]=[
    'png, jpg, jpeg, webp'], ext_case_sensitive: bool=False):
    self.__images__: List[str] = []
    self.__labels__: List[int] = []
    if isinstance(root, str) or isinstance(root, os.PathLike) or isinstance(
        root, pathlib.PurePath):
        for dir, label in dir_label_map.items():
            appended_list: List[str] = list_all_images(os.path.join(root,
                dir), image_exts=image_exts, ext_case_sensitive=
                ext_case_sensitive)
            self.__images__ += appended_list
            self.__labels__ += [label] * len(appended_list)
    else:
        raise TypeError(
            f'The arguement image should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}'
            )
    self.__vmin: float = vmin
    self.__vmax: float = vmax
    self.__generator__ = set_generator(generator=generator)
    self.__trans__ = transforms.Compose([transforms.Resize(size=size),
        transforms.ToTensor(), transforms.ConvertImageDtype(torch.float),
        transforms.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1,
        vmin_out=self.__vmin, vmax_out=self.__vmax, x=x))])
