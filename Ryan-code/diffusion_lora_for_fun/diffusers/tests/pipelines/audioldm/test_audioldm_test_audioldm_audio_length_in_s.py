def test_audioldm_audio_length_in_s(self):
    device = 'cpu'
    components = self.get_dummy_components()
    audioldm_pipe = AudioLDMPipeline(**components)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    vocoder_sampling_rate = audioldm_pipe.vocoder.config.sampling_rate
    inputs = self.get_dummy_inputs(device)
    output = audioldm_pipe(audio_length_in_s=0.016, **inputs)
    audio = output.audios[0]
    assert audio.ndim == 1
    assert len(audio) / vocoder_sampling_rate == 0.016
    output = audioldm_pipe(audio_length_in_s=0.032, **inputs)
    audio = output.audios[0]
    assert audio.ndim == 1
    assert len(audio) / vocoder_sampling_rate == 0.032
