def test_musicldm_ddim(self):
    device = 'cpu'
    components = self.get_dummy_components()
    musicldm_pipe = MusicLDMPipeline(**components)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = musicldm_pipe(**inputs)
    audio = output.audios[0]
    assert audio.ndim == 1
    assert len(audio) == 256
    audio_slice = audio[:10]
    expected_slice = np.array([-0.0027, -0.0036, -0.0037, -0.002, -0.0035, 
        -0.0019, -0.0037, -0.002, -0.0038, -0.0019])
    assert np.abs(audio_slice - expected_slice).max() < 0.0001
