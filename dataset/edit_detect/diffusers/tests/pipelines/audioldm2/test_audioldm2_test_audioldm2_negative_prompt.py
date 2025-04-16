def test_audioldm2_negative_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = PNDMScheduler(skip_prk_steps=True)
    audioldm_pipe = AudioLDM2Pipeline(**components)
    audioldm_pipe = audioldm_pipe.to(device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    negative_prompt = 'egg cracking'
    output = audioldm_pipe(**inputs, negative_prompt=negative_prompt)
    audio = output.audios[0]
    assert audio.ndim == 1
    assert len(audio) == 256
    audio_slice = audio[:10]
    expected_slice = np.array([0.0025, 0.0018, 0.0018, -0.0023, -0.0026, -
        0.002, -0.0026, -0.0021, -0.0027, -0.002])
    assert np.abs(audio_slice - expected_slice).max() < 0.0001
