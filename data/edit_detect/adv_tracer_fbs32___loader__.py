def __loader__(self, data: Union[torch.Tensor, Image.Image, str, os.
    PathLike, pathlib.PurePath]):
    trans = [transforms.Resize(size=self.__size__), transforms.
        ConvertImageDtype(torch.float), self.__forwar_reverse_fn__()[0]]
    if isinstance(data, Image.Image):
        trans = [transforms.ToTensor()] + trans
    elif isinstance(data, str) or isinstance(data, os.PathLike) or isinstance(
        data, pathlib.PurePath):
        opening_image = transforms.Lambda(lambda x: Image.open(x).convert(
            'RGB'))
        trans = [opening_image, transforms.ToTensor()] + trans
    else:
        raise TypeError(
            f'The arguement data should be torch.Tensor, Image.Image, str, os.PathLike, or pathlib.PurePath, not {type(image)}'
            )
    return transforms.Compose(trans)(data)
