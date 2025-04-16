def test_musicldm_vocoder_model_in_dim(self):
    components = self.get_dummy_components()
    musicldm_pipe = MusicLDMPipeline(**components)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe.set_progress_bar_config(disable=None)
    prompt = ['hey']
    output = musicldm_pipe(prompt, num_inference_steps=1)
    audio_shape = output.audios.shape
    assert audio_shape == (1, 256)
    config = musicldm_pipe.vocoder.config
    config.model_in_dim *= 2
    musicldm_pipe.vocoder = SpeechT5HifiGan(config).to(torch_device)
    output = musicldm_pipe(prompt, num_inference_steps=1)
    audio_shape = output.audios.shape
    assert audio_shape == (1, 256)
