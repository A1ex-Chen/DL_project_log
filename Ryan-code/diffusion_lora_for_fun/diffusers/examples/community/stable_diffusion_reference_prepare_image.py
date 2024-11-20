def prepare_image(self, image: Union[torch.Tensor, PIL.Image.Image, List[
    Union[torch.Tensor, PIL.Image.Image]]], width: int, height: int,
    batch_size: int, num_images_per_prompt: int, device: torch.device,
    dtype: torch.dtype, do_classifier_free_guidance: bool=False, guess_mode:
    bool=False) ->torch.Tensor:
    """
        Prepares the input image for processing.

        Args:
            image (torch.Tensor or PIL.Image.Image or list): The input image(s).
            width (int): The desired width of the image.
            height (int): The desired height of the image.
            batch_size (int): The batch size for processing.
            num_images_per_prompt (int): The number of images per prompt.
            device (torch.device): The device to use for processing.
            dtype (torch.dtype): The data type of the image.
            do_classifier_free_guidance (bool, optional): Whether to perform classifier-free guidance. Defaults to False.
            guess_mode (bool, optional): Whether to use guess mode. Defaults to False.

        Returns:
            torch.Tensor: The prepared image for processing.
        """
    if not isinstance(image, torch.Tensor):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        if isinstance(image[0], PIL.Image.Image):
            images = []
            for image_ in image:
                image_ = image_.convert('RGB')
                image_ = image_.resize((width, height), resample=
                    PIL_INTERPOLATION['lanczos'])
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)
            image = images
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
    image_batch_size = image.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        repeat_by = num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)
    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)
    return image
