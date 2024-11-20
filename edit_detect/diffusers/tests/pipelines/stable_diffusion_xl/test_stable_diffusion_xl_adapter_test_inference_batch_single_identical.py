def test_inference_batch_single_identical(self, batch_size=3,
    test_max_difference=None, test_mean_pixel_difference=None,
    relax_max_difference=False, expected_max_diff=0.002,
    additional_params_copy_to_batched_inputs=['num_inference_steps']):
    if test_max_difference is None:
        test_max_difference = torch_device != 'mps'
    if test_mean_pixel_difference is None:
        test_mean_pixel_difference = torch_device != 'mps'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    logger = logging.get_logger(pipe.__module__)
    logger.setLevel(level=diffusers.logging.FATAL)
    batched_inputs = {}
    batch_size = batch_size
    for name, value in inputs.items():
        if name in self.batch_params:
            if name == 'prompt':
                len_prompt = len(value)
                batched_inputs[name] = [value[:len_prompt // i] for i in
                    range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * 'very long'
            elif name == 'image':
                batched_images = []
                for image in value:
                    batched_images.append(batch_size * [image])
                batched_inputs[name] = batched_images
            else:
                batched_inputs[name] = batch_size * [value]
        elif name == 'batch_size':
            batched_inputs[name] = batch_size
        elif name == 'generator':
            batched_inputs[name] = [self.get_generator(i) for i in range(
                batch_size)]
        else:
            batched_inputs[name] = value
    for arg in additional_params_copy_to_batched_inputs:
        batched_inputs[arg] = inputs[arg]
    output_batch = pipe(**batched_inputs)
    assert output_batch[0].shape[0] == batch_size
    inputs['generator'] = self.get_generator(0)
    output = pipe(**inputs)
    logger.setLevel(level=diffusers.logging.WARNING)
    if test_max_difference:
        if relax_max_difference:
            diff = np.abs(output_batch[0][0] - output[0][0])
            diff = diff.flatten()
            diff.sort()
            max_diff = np.median(diff[-5:])
        else:
            max_diff = np.abs(output_batch[0][0] - output[0][0]).max()
        assert max_diff < expected_max_diff
    if test_mean_pixel_difference:
        assert_mean_pixel_difference(output_batch[0][0], output[0][0])
