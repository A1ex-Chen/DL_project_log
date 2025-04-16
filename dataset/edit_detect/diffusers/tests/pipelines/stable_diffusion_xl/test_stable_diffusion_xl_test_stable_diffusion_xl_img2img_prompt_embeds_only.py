def test_stable_diffusion_xl_img2img_prompt_embeds_only(self):
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    inputs = self.get_dummy_inputs(generator_device)
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = sd_pipe(**inputs)
    image_slice_1 = output.images[0, -3:, -3:, -1]
    generator_device = 'cpu'
    inputs = self.get_dummy_inputs(generator_device)
    prompt = 3 * [inputs.pop('prompt')]
    prompt_embeds, _, pooled_prompt_embeds, _ = sd_pipe.encode_prompt(prompt)
    output = sd_pipe(**inputs, prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds)
    image_slice_2 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max(
        ) < 0.0001
