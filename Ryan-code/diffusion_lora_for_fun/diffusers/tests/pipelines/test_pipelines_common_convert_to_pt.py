def convert_to_pt(image):
    if isinstance(image, torch.Tensor):
        input_image = image
    elif isinstance(image, np.ndarray):
        input_image = VaeImageProcessor.numpy_to_pt(image)
    elif isinstance(image, PIL.Image.Image):
        input_image = VaeImageProcessor.pil_to_numpy(image)
        input_image = VaeImageProcessor.numpy_to_pt(input_image)
    else:
        raise ValueError(f'unsupported input_image_type {type(image)}')
    return input_image
