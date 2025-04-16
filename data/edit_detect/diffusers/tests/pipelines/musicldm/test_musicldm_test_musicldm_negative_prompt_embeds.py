def test_musicldm_negative_prompt_embeds(self):
    components = self.get_dummy_components()
    musicldm_pipe = MusicLDMPipeline(**components)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    negative_prompt = 3 * ['this is a negative prompt']
    inputs['negative_prompt'] = negative_prompt
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = musicldm_pipe(**inputs)
    audio_1 = output.audios[0]
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 3 * [inputs.pop('prompt')]
    embeds = []
    for p in [prompt, negative_prompt]:
        text_inputs = musicldm_pipe.tokenizer(p, padding='max_length',
            max_length=musicldm_pipe.tokenizer.model_max_length, truncation
            =True, return_tensors='pt')
        text_inputs = text_inputs['input_ids'].to(torch_device)
        text_embeds = musicldm_pipe.text_encoder.get_text_features(text_inputs)
        embeds.append(text_embeds)
    inputs['prompt_embeds'], inputs['negative_prompt_embeds'] = embeds
    output = musicldm_pipe(**inputs)
    audio_2 = output.audios[0]
    assert np.abs(audio_1 - audio_2).max() < 0.01
