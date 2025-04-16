def test_audioldm_negative_prompt_embeds(self):
    components = self.get_dummy_components()
    audioldm_pipe = AudioLDMPipeline(**components)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    negative_prompt = 3 * ['this is a negative prompt']
    inputs['negative_prompt'] = negative_prompt
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = audioldm_pipe(**inputs)
    audio_1 = output.audios[0]
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 3 * [inputs.pop('prompt')]
    embeds = []
    for p in [prompt, negative_prompt]:
        text_inputs = audioldm_pipe.tokenizer(p, padding='max_length',
            max_length=audioldm_pipe.tokenizer.model_max_length, truncation
            =True, return_tensors='pt')
        text_inputs = text_inputs['input_ids'].to(torch_device)
        text_embeds = audioldm_pipe.text_encoder(text_inputs)
        text_embeds = text_embeds.text_embeds
        text_embeds = F.normalize(text_embeds, dim=-1)
        embeds.append(text_embeds)
    inputs['prompt_embeds'], inputs['negative_prompt_embeds'] = embeds
    output = audioldm_pipe(**inputs)
    audio_2 = output.audios[0]
    assert np.abs(audio_1 - audio_2).max() < 0.01
