def test_audioldm2(self):
    audioldm_pipe = AudioLDM2Pipeline.from_pretrained('cvssp/audioldm2')
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 25
    audio = audioldm_pipe(**inputs).audios[0]
    assert audio.ndim == 1
    assert len(audio) == 81952
    audio_slice = audio[17275:17285]
    expected_slice = np.array([0.0791, 0.0666, 0.1158, 0.1227, 0.1171, -
        0.288, -0.194, -0.0283, -0.0126, 0.1127])
    max_diff = np.abs(expected_slice - audio_slice).max()
    assert max_diff < 0.001
