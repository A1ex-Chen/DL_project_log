def test_audioldm2_prompt_embeds(self):
    components = self.get_dummy_components()
    audioldm_pipe = AudioLDM2Pipeline(**components)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = audioldm_pipe(**inputs)
    audio_1 = output.audios[0]
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 3 * [inputs.pop('prompt')]
    text_inputs = audioldm_pipe.tokenizer(prompt, padding='max_length',
        max_length=audioldm_pipe.tokenizer.model_max_length, truncation=
        True, return_tensors='pt')
    text_inputs = text_inputs['input_ids'].to(torch_device)
    clap_prompt_embeds = audioldm_pipe.text_encoder.get_text_features(
        text_inputs)
    clap_prompt_embeds = clap_prompt_embeds[:, None, :]
    text_inputs = audioldm_pipe.tokenizer_2(prompt, padding='max_length',
        max_length=True, truncation=True, return_tensors='pt')
    text_inputs = text_inputs['input_ids'].to(torch_device)
    t5_prompt_embeds = audioldm_pipe.text_encoder_2(text_inputs)
    t5_prompt_embeds = t5_prompt_embeds[0]
    projection_embeds = audioldm_pipe.projection_model(clap_prompt_embeds,
        t5_prompt_embeds)[0]
    generated_prompt_embeds = audioldm_pipe.generate_language_model(
        projection_embeds, max_new_tokens=8)
    inputs['prompt_embeds'] = t5_prompt_embeds
    inputs['generated_prompt_embeds'] = generated_prompt_embeds
    output = audioldm_pipe(**inputs)
    audio_2 = output.audios[0]
    assert np.abs(audio_1 - audio_2).max() < 0.01
