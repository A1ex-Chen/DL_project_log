def get_dummy_inputs_by_type(self, device, seed=0, input_image_type='pt',
    output_type='np'):
    inputs = self.get_dummy_inputs(device, seed)

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

    def convert_pt_to_type(image, input_image_type):
        if input_image_type == 'pt':
            input_image = image
        elif input_image_type == 'np':
            input_image = VaeImageProcessor.pt_to_numpy(image)
        elif input_image_type == 'pil':
            input_image = VaeImageProcessor.pt_to_numpy(image)
            input_image = VaeImageProcessor.numpy_to_pil(input_image)
        else:
            raise ValueError(
                f'unsupported input_image_type {input_image_type}.')
        return input_image
    for image_param in self.image_params:
        if image_param in inputs.keys():
            inputs[image_param] = convert_pt_to_type(convert_to_pt(inputs[
                image_param]).to(device), input_image_type)
    inputs['output_type'] = output_type
    return inputs
