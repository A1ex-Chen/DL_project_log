def test_audioldm2_lms(self):
    audioldm_pipe = AudioLDM2Pipeline.from_pretrained('cvssp/audioldm2')
    audioldm_pipe.scheduler = LMSDiscreteScheduler.from_config(audioldm_pipe
        .scheduler.config)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    audio = audioldm_pipe(**inputs).audios[0]
    assert audio.ndim == 1
    assert len(audio) == 81952
    audio_slice = audio[31390:31400]
    expected_slice = np.array([-0.1318, -0.0577, 0.0446, -0.0573, 0.0659, 
        0.1074, -0.26, 0.008, -0.219, -0.4301])
    max_diff = np.abs(expected_slice - audio_slice).max()
    assert max_diff < 0.001
