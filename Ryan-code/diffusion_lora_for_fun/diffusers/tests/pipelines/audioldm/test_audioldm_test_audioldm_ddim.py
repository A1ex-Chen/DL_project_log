def test_audioldm_ddim(self):
    device = 'cpu'
    components = self.get_dummy_components()
    audioldm_pipe = AudioLDMPipeline(**components)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = audioldm_pipe(**inputs)
    audio = output.audios[0]
    assert audio.ndim == 1
    assert len(audio) == 256
    audio_slice = audio[:10]
    expected_slice = np.array([-0.005, 0.005, -0.006, 0.0033, -0.0026, 
        0.0033, -0.0027, 0.0033, -0.0028, 0.0033])
    assert np.abs(audio_slice - expected_slice).max() < 0.01
