def check_inputs(self, prompt, image, mask_image,
    controlnet_conditioning_image, height, width, callback_steps,
    negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
    strength=None):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
            )
    if callback_steps is None or callback_steps is not None and (not
        isinstance(callback_steps, int) or callback_steps <= 0):
        raise ValueError(
            f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
            )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
            )
    elif prompt is not None and (not isinstance(prompt, str) and not
        isinstance(prompt, list)):
        raise ValueError(
            f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )
    controlnet_cond_image_is_pil = isinstance(controlnet_conditioning_image,
        PIL.Image.Image)
    controlnet_cond_image_is_tensor = isinstance(controlnet_conditioning_image,
        torch.Tensor)
    controlnet_cond_image_is_pil_list = isinstance(
        controlnet_conditioning_image, list) and isinstance(
        controlnet_conditioning_image[0], PIL.Image.Image)
    controlnet_cond_image_is_tensor_list = isinstance(
        controlnet_conditioning_image, list) and isinstance(
        controlnet_conditioning_image[0], torch.Tensor)
    if (not controlnet_cond_image_is_pil and not
        controlnet_cond_image_is_tensor and not
        controlnet_cond_image_is_pil_list and not
        controlnet_cond_image_is_tensor_list):
        raise TypeError(
            'image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors'
            )
    if controlnet_cond_image_is_pil:
        controlnet_cond_image_batch_size = 1
    elif controlnet_cond_image_is_tensor:
        controlnet_cond_image_batch_size = controlnet_conditioning_image.shape[
            0]
    elif controlnet_cond_image_is_pil_list:
        controlnet_cond_image_batch_size = len(controlnet_conditioning_image)
    elif controlnet_cond_image_is_tensor_list:
        controlnet_cond_image_batch_size = len(controlnet_conditioning_image)
    if prompt is not None and isinstance(prompt, str):
        prompt_batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        prompt_batch_size = len(prompt)
    elif prompt_embeds is not None:
        prompt_batch_size = prompt_embeds.shape[0]
    if (controlnet_cond_image_batch_size != 1 and 
        controlnet_cond_image_batch_size != prompt_batch_size):
        raise ValueError(
            f'If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {controlnet_cond_image_batch_size}, prompt batch size: {prompt_batch_size}'
            )
    if isinstance(image, torch.Tensor) and not isinstance(mask_image, torch
        .Tensor):
        raise TypeError(
            'if `image` is a tensor, `mask_image` must also be a tensor')
    if isinstance(image, PIL.Image.Image) and not isinstance(mask_image,
        PIL.Image.Image):
        raise TypeError(
            'if `image` is a PIL image, `mask_image` must also be a PIL image')
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 and image.ndim != 4:
            raise ValueError('`image` must have 3 or 4 dimensions')
        if (mask_image.ndim != 2 and mask_image.ndim != 3 and mask_image.
            ndim != 4):
            raise ValueError('`mask_image` must have 2, 3, or 4 dimensions')
        if image.ndim == 3:
            image_batch_size = 1
            image_channels, image_height, image_width = image.shape
        elif image.ndim == 4:
            image_batch_size, image_channels, image_height, image_width = (
                image.shape)
        if mask_image.ndim == 2:
            mask_image_batch_size = 1
            mask_image_channels = 1
            mask_image_height, mask_image_width = mask_image.shape
        elif mask_image.ndim == 3:
            mask_image_channels = 1
            mask_image_batch_size, mask_image_height, mask_image_width = (
                mask_image.shape)
        elif mask_image.ndim == 4:
            (mask_image_batch_size, mask_image_channels, mask_image_height,
                mask_image_width) = mask_image.shape
        if image_channels != 3:
            raise ValueError('`image` must have 3 channels')
        if mask_image_channels != 1:
            raise ValueError('`mask_image` must have 1 channel')
        if image_batch_size != mask_image_batch_size:
            raise ValueError(
                '`image` and `mask_image` mush have the same batch sizes')
        if (image_height != mask_image_height or image_width !=
            mask_image_width):
            raise ValueError(
                '`image` and `mask_image` must have the same height and width dimensions'
                )
        if image.min() < -1 or image.max() > 1:
            raise ValueError('`image` should be in range [-1, 1]')
        if mask_image.min() < 0 or mask_image.max() > 1:
            raise ValueError('`mask_image` should be in range [0, 1]')
    else:
        mask_image_channels = 1
        image_channels = 3
    single_image_latent_channels = self.vae.config.latent_channels
    total_latent_channels = (single_image_latent_channels * 2 +
        mask_image_channels)
    if total_latent_channels != self.unet.config.in_channels:
        raise ValueError(
            f'The config of `pipeline.unet` expects {self.unet.config.in_channels} but received non inpainting latent channels: {single_image_latent_channels}, mask channels: {mask_image_channels}, and masked image channels: {single_image_latent_channels}. Please verify the config of `pipeline.unet` and the `mask_image` and `image` inputs.'
            )
    if strength < 0 or strength > 1:
        raise ValueError(
            f'The value of strength should in [0.0, 1.0] but is {strength}')
