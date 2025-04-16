def test_audioldm_num_waveforms_per_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = PNDMScheduler(skip_prk_steps=True)
    audioldm_pipe = AudioLDMPipeline(**components)
    audioldm_pipe = audioldm_pipe.to(device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    prompt = 'A hammer hitting a wooden surface'
    audios = audioldm_pipe(prompt, num_inference_steps=2).audios
    assert audios.shape == (1, 256)
    batch_size = 2
    audios = audioldm_pipe([prompt] * batch_size, num_inference_steps=2).audios
    assert audios.shape == (batch_size, 256)
    num_waveforms_per_prompt = 2
    audios = audioldm_pipe(prompt, num_inference_steps=2,
        num_waveforms_per_prompt=num_waveforms_per_prompt).audios
    assert audios.shape == (num_waveforms_per_prompt, 256)
    batch_size = 2
    audios = audioldm_pipe([prompt] * batch_size, num_inference_steps=2,
        num_waveforms_per_prompt=num_waveforms_per_prompt).audios
    assert audios.shape == (batch_size * num_waveforms_per_prompt, 256)
