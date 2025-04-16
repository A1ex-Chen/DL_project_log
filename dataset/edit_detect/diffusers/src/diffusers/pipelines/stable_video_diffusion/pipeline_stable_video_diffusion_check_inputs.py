def check_inputs(self, image, height, width):
    if not isinstance(image, torch.Tensor) and not isinstance(image, PIL.
        Image.Image) and not isinstance(image, list):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is {type(image)}'
            )
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
