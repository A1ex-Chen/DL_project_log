def _resize_bilinear(image: Union[torch.Tensor, List[torch.Tensor], PIL.
    Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]):
    image = _convert_pt_to_pil(image)
    if isinstance(image, list):
        image = [u.resize(resolution, PIL.Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, PIL.Image.BILINEAR)
    return image
