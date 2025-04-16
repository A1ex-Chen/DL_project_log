def __load_images__(self, images: Union[List[Union[torch.Tensor, Image.
    Image, str, os.PathLike, pathlib.PurePath]], torch.Tensor, Image.Image,
    str, os.PathLike, pathlib.PurePath], num: int, epsilon_scale: float
    ) ->torch.Tensor:
    if isinstance(images, torch.Tensor):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
    elif isinstance(images, np.ndarray):
        if len(images.shape) == 3:
            images = torch.from_numpy(images).unsqueeze(0)
    elif not isinstance(images, list):
        images = [images]
    img_ls = []
    for img in images:
        processed_img: torch.Tensor = self.__process_img__(image=img)
        img_ls.append(processed_img)
    return torch.stack(img_ls)
