def test_stable_diffusion_prompt_embeds(self):
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = sd_pipe(**inputs)
    image_slice_1 = output.images[0, -3:, -3:, -1]
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 3 * [inputs.pop('prompt')]
    text_inputs = sd_pipe.tokenizer(prompt, padding='max_length',
        max_length=sd_pipe.tokenizer.model_max_length, truncation=True,
        return_tensors='pt')
    text_inputs = text_inputs['input_ids'].to(torch_device)
    prompt_embeds = sd_pipe.text_encoder(text_inputs)[0]
    inputs['prompt_embeds'] = prompt_embeds
    output = sd_pipe(**inputs)
    image_slice_2 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max(
        ) < 0.0001
