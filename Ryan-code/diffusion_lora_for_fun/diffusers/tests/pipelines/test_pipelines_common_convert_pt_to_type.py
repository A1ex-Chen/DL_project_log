def convert_pt_to_type(image, input_image_type):
    if input_image_type == 'pt':
        input_image = image
    elif input_image_type == 'np':
        input_image = VaeImageProcessor.pt_to_numpy(image)
    elif input_image_type == 'pil':
        input_image = VaeImageProcessor.pt_to_numpy(image)
        input_image = VaeImageProcessor.numpy_to_pil(input_image)
    else:
        raise ValueError(f'unsupported input_image_type {input_image_type}.')
    return input_image
