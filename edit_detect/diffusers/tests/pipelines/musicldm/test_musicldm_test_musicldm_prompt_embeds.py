def test_musicldm_prompt_embeds(self):
    components = self.get_dummy_components()
    musicldm_pipe = MusicLDMPipeline(**components)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = musicldm_pipe(**inputs)
    audio_1 = output.audios[0]
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 3 * [inputs.pop('prompt')]
    text_inputs = musicldm_pipe.tokenizer(prompt, padding='max_length',
        max_length=musicldm_pipe.tokenizer.model_max_length, truncation=
        True, return_tensors='pt')
    text_inputs = text_inputs['input_ids'].to(torch_device)
    prompt_embeds = musicldm_pipe.text_encoder.get_text_features(text_inputs)
    inputs['prompt_embeds'] = prompt_embeds
    output = musicldm_pipe(**inputs)
    audio_2 = output.audios[0]
    assert np.abs(audio_1 - audio_2).max() < 0.01
