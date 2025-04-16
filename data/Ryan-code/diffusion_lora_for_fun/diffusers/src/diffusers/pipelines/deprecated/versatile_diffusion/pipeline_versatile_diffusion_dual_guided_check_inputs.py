def check_inputs(self, prompt, image, height, width, callback_steps):
    if not isinstance(prompt, str) and not isinstance(prompt, PIL.Image.Image
        ) and not isinstance(prompt, list):
        raise ValueError(
            f'`prompt` has to be of type `str` `PIL.Image` or `list` but is {type(prompt)}'
            )
    if not isinstance(image, str) and not isinstance(image, PIL.Image.Image
        ) and not isinstance(image, list):
        raise ValueError(
            f'`image` has to be of type `str` `PIL.Image` or `list` but is {type(image)}'
            )
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
