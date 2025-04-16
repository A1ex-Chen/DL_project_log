def test_stable_diffusion_xl_prompt_embeds(self):
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['prompt'] = 2 * [inputs['prompt']]
    inputs['num_images_per_prompt'] = 2
    output = sd_pipe(**inputs)
    image_slice_1 = output.images[0, -3:, -3:, -1]
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 2 * [inputs.pop('prompt')]
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = sd_pipe.encode_prompt(prompt)
    output = sd_pipe(**inputs, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds
        =pooled_prompt_embeds, negative_pooled_prompt_embeds=
        negative_pooled_prompt_embeds)
    image_slice_2 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max(
        ) < 0.0001
