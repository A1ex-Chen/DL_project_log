def test_inference_multiple_prompt_input(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionControlNetPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt'], inputs['prompt']]
    inputs['image'] = [inputs['image'], inputs['image']]
    output = sd_pipe(**inputs)
    image = output.images
    assert image.shape == (2, 64, 64, 3)
    image_1, image_2 = image
    assert np.sum(np.abs(image_1 - image_2)) > 0.001
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt'], inputs['prompt']]
    output_1 = sd_pipe(**inputs)
    assert np.abs(image - output_1.images).max() < 0.001
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt'], inputs['prompt'], inputs['prompt'
        ], inputs['prompt']]
    inputs['image'] = [inputs['image'], inputs['image'], inputs['image'],
        inputs['image']]
    output_2 = sd_pipe(**inputs)
    image = output_2.images
    assert image.shape == (4, 64, 64, 3)
