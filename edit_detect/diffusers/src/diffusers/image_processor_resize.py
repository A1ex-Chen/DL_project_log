def resize(self, image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
    height: int, width: int, resize_mode: str='default') ->Union[PIL.Image.
    Image, np.ndarray, torch.Tensor]:
    """
        Resize image.

        Args:
            image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
                The image input, can be a PIL image, numpy array or pytorch tensor.
            height (`int`):
                The height to resize to.
            width (`int`):
                The width to resize to.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode to use, can be one of `default` or `fill`. If `default`, will resize the image to fit
                within the specified width and height, and it may not maintaining the original aspect ratio. If `fill`,
                will resize the image to fit within the specified width and height, maintaining the aspect ratio, and
                then center the image within the dimensions, filling empty with data from image. If `crop`, will resize
                the image to fit within the specified width and height, maintaining the aspect ratio, and then center
                the image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.

        Returns:
            `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
                The resized image.
        """
    if resize_mode != 'default' and not isinstance(image, PIL.Image.Image):
        raise ValueError(
            f'Only PIL image input is supported for resize_mode {resize_mode}')
    if isinstance(image, PIL.Image.Image):
        if resize_mode == 'default':
            image = image.resize((width, height), resample=
                PIL_INTERPOLATION[self.config.resample])
        elif resize_mode == 'fill':
            image = self._resize_and_fill(image, width, height)
        elif resize_mode == 'crop':
            image = self._resize_and_crop(image, width, height)
        else:
            raise ValueError(f'resize_mode {resize_mode} is not supported')
    elif isinstance(image, torch.Tensor):
        image = torch.nn.functional.interpolate(image, size=(height, width))
    elif isinstance(image, np.ndarray):
        image = self.numpy_to_pt(image)
        image = torch.nn.functional.interpolate(image, size=(height, width))
        image = self.pt_to_numpy(image)
    return image
