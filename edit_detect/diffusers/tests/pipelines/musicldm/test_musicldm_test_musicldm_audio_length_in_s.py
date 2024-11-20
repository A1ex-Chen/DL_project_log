def test_musicldm_audio_length_in_s(self):
    device = 'cpu'
    components = self.get_dummy_components()
    musicldm_pipe = MusicLDMPipeline(**components)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe.set_progress_bar_config(disable=None)
    vocoder_sampling_rate = musicldm_pipe.vocoder.config.sampling_rate
    inputs = self.get_dummy_inputs(device)
    output = musicldm_pipe(audio_length_in_s=0.016, **inputs)
    audio = output.audios[0]
    assert audio.ndim == 1
    assert len(audio) / vocoder_sampling_rate == 0.016
    output = musicldm_pipe(audio_length_in_s=0.032, **inputs)
    audio = output.audios[0]
    assert audio.ndim == 1
    assert len(audio) / vocoder_sampling_rate == 0.032
