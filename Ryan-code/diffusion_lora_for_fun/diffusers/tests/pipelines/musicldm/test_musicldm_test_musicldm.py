def test_musicldm(self):
    musicldm_pipe = MusicLDMPipeline.from_pretrained('cvssp/musicldm')
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 25
    audio = musicldm_pipe(**inputs).audios[0]
    assert audio.ndim == 1
    assert len(audio) == 81952
    audio_slice = audio[8680:8690]
    expected_slice = np.array([-0.1042, -0.1068, -0.1235, -0.1387, -0.1428,
        -0.136, -0.1213, -0.1097, -0.0967, -0.0945])
    max_diff = np.abs(expected_slice - audio_slice).max()
    assert max_diff < 0.001
