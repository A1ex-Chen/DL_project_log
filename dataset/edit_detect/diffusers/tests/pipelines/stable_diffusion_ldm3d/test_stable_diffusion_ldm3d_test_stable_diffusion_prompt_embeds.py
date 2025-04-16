def test_stable_diffusion_prompt_embeds(self):
    components = self.get_dummy_components()
    ldm3d_pipe = StableDiffusionLDM3DPipeline(**components)
    ldm3d_pipe = ldm3d_pipe.to(torch_device)
    ldm3d_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = ldm3d_pipe(**inputs)
    rgb_slice_1, depth_slice_1 = output.rgb, output.depth
    rgb_slice_1 = rgb_slice_1[0, -3:, -3:, -1]
    depth_slice_1 = depth_slice_1[0, -3:, -1]
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 3 * [inputs.pop('prompt')]
    text_inputs = ldm3d_pipe.tokenizer(prompt, padding='max_length',
        max_length=ldm3d_pipe.tokenizer.model_max_length, truncation=True,
        return_tensors='pt')
    text_inputs = text_inputs['input_ids'].to(torch_device)
    prompt_embeds = ldm3d_pipe.text_encoder(text_inputs)[0]
    inputs['prompt_embeds'] = prompt_embeds
    output = ldm3d_pipe(**inputs)
    rgb_slice_2, depth_slice_2 = output.rgb, output.depth
    rgb_slice_2 = rgb_slice_2[0, -3:, -3:, -1]
    depth_slice_2 = depth_slice_2[0, -3:, -1]
    assert np.abs(rgb_slice_1.flatten() - rgb_slice_2.flatten()).max() < 0.0001
    assert np.abs(depth_slice_1.flatten() - depth_slice_2.flatten()).max(
        ) < 0.0001
