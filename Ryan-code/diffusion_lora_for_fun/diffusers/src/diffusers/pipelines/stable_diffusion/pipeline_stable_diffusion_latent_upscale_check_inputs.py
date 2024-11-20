def check_inputs(self, prompt, image, callback_steps):
    if not isinstance(prompt, str) and not isinstance(prompt, list):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if not isinstance(image, torch.Tensor) and not isinstance(image, PIL.
        Image.Image) and not isinstance(image, list):
        raise ValueError(
            f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or `list` but is {type(image)}'
            )
    if isinstance(image, list) or isinstance(image, torch.Tensor):
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        if isinstance(image, list):
            image_batch_size = len(image)
        else:
            image_batch_size = image.shape[0] if image.ndim == 4 else 1
        if batch_size != image_batch_size:
            raise ValueError(
                f'`prompt` has batch size {batch_size} and `image` has batch size {image_batch_size}. Please make sure that passed `prompt` matches the batch size of `image`.'
                )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
