def test_musicldm_lms(self):
    musicldm_pipe = MusicLDMPipeline.from_pretrained('cvssp/musicldm')
    musicldm_pipe.scheduler = LMSDiscreteScheduler.from_config(musicldm_pipe
        .scheduler.config)
    musicldm_pipe = musicldm_pipe.to(torch_device)
    musicldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    audio = musicldm_pipe(**inputs).audios[0]
    assert audio.ndim == 1
    assert len(audio) == 81952
    audio_slice = audio[58020:58030]
    expected_slice = np.array([0.3592, 0.3477, 0.4084, 0.4665, 0.5048, 
        0.5891, 0.6461, 0.5579, 0.4595, 0.4403])
    max_diff = np.abs(expected_slice - audio_slice).max()
    assert max_diff < 0.001
