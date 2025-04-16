def test_audioldm2_vocoder_model_in_dim(self):
    components = self.get_dummy_components()
    audioldm_pipe = AudioLDM2Pipeline(**components)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    prompt = ['hey']
    output = audioldm_pipe(prompt, num_inference_steps=1)
    audio_shape = output.audios.shape
    assert audio_shape == (1, 256)
    config = audioldm_pipe.vocoder.config
    config.model_in_dim *= 2
    audioldm_pipe.vocoder = SpeechT5HifiGan(config).to(torch_device)
    output = audioldm_pipe(prompt, num_inference_steps=1)
    audio_shape = output.audios.shape
    assert audio_shape == (1, 256)
