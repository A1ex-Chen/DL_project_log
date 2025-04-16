def check_image(self, image, prompt, prompt_embeds):
    image_is_pil = isinstance(image, PIL.Image.Image)
    image_is_tensor = isinstance(image, torch.Tensor)
    image_is_np = isinstance(image, np.ndarray)
    image_is_pil_list = isinstance(image, list) and isinstance(image[0],
        PIL.Image.Image)
    image_is_tensor_list = isinstance(image, list) and isinstance(image[0],
        torch.Tensor)
    image_is_np_list = isinstance(image, list) and isinstance(image[0], np.
        ndarray)
    if (not image_is_pil and not image_is_tensor and not image_is_np and 
        not image_is_pil_list and not image_is_tensor_list and not
        image_is_np_list):
        raise TypeError(
            f'image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}'
            )
    if image_is_pil:
        image_batch_size = 1
    else:
        image_batch_size = len(image)
    if prompt is not None and isinstance(prompt, str):
        prompt_batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        prompt_batch_size = len(prompt)
    elif prompt_embeds is not None:
        prompt_batch_size = prompt_embeds.shape[0]
    if image_batch_size != 1 and image_batch_size != prompt_batch_size:
        raise ValueError(
            f'If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}'
            )
