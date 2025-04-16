def _default_height_width(self, height: Optional[int], width: Optional[int],
    image: Union[PIL.Image.Image, torch.Tensor, List[PIL.Image.Image]]
    ) ->Tuple[int, int]:
    """
        Calculate the default height and width for the given image.

        Args:
            height (int or None): The desired height of the image. If None, the height will be determined based on the input image.
            width (int or None): The desired width of the image. If None, the width will be determined based on the input image.
            image (PIL.Image.Image or torch.Tensor or list[PIL.Image.Image]): The input image or a list of images.

        Returns:
            Tuple[int, int]: A tuple containing the calculated height and width.

        """
    while isinstance(image, list):
        image = image[0]
    if height is None:
        if isinstance(image, PIL.Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[2]
        height = height // 8 * 8
    if width is None:
        if isinstance(image, PIL.Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[3]
        width = width // 8 * 8
    return height, width
