def test_stable_diffusion_xl_multi_prompts(self):
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components).to(torch_device)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 5
    output = sd_pipe(**inputs)
    image_slice_1 = output.images[0, -3:, -3:, -1]
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 5
    inputs['prompt_2'] = inputs['prompt']
    output = sd_pipe(**inputs)
    image_slice_2 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max(
        ) < 0.0001
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 5
    inputs['prompt_2'] = 'different prompt'
    output = sd_pipe(**inputs)
    image_slice_3 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max(
        ) > 0.0001
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 5
    inputs['negative_prompt'] = 'negative prompt'
    output = sd_pipe(**inputs)
    image_slice_1 = output.images[0, -3:, -3:, -1]
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 5
    inputs['negative_prompt'] = 'negative prompt'
    inputs['negative_prompt_2'] = inputs['negative_prompt']
    output = sd_pipe(**inputs)
    image_slice_2 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max(
        ) < 0.0001
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 5
    inputs['negative_prompt'] = 'negative prompt'
    inputs['negative_prompt_2'] = 'different negative prompt'
    output = sd_pipe(**inputs)
    image_slice_3 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max(
        ) > 0.0001
