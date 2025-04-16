def _center_crop_wide(image: Union[torch.Tensor, List[torch.Tensor], PIL.
    Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]):
    image = _convert_pt_to_pil(image)
    if isinstance(image, list):
        scale = min(image[0].size[0] / resolution[0], image[0].size[1] /
            resolution[1])
        image = [u.resize((round(u.width // scale), round(u.height // scale
            )), resample=PIL.Image.BOX) for u in image]
        x1 = (image[0].width - resolution[0]) // 2
        y1 = (image[0].height - resolution[1]) // 2
        image = [u.crop((x1, y1, x1 + resolution[0], y1 + resolution[1])) for
            u in image]
        return image
    else:
        scale = min(image.size[0] / resolution[0], image.size[1] /
            resolution[1])
        image = image.resize((round(image.width // scale), round(image.
            height // scale)), resample=PIL.Image.BOX)
        x1 = (image.width - resolution[0]) // 2
        y1 = (image.height - resolution[1]) // 2
        image = image.crop((x1, y1, x1 + resolution[0], y1 + resolution[1]))
        return image
