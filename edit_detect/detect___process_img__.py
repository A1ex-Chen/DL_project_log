def __process_img__(self, image: Union[torch.Tensor, Image.Image, str, os.
    PathLike, pathlib.PurePath]):
    if isinstance(image, torch.Tensor):
        return self.__trans__(image)
    elif isinstance(image, Image.Image):
        return self.__trans__(transforms.ToTensor()(image))
    elif isinstance(image, str) or isinstance(image, os.PathLike
        ) or isinstance(image, pathlib.PurePath):
        return self.__trans__(transforms.ToTensor()(Image.open(image)))
    else:
        raise TypeError(
            f'The arguement image should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}'
            )
