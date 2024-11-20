def test_audioldm_lms(self):
    audioldm_pipe = AudioLDMPipeline.from_pretrained('cvssp/audioldm')
    audioldm_pipe.scheduler = LMSDiscreteScheduler.from_config(audioldm_pipe
        .scheduler.config)
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    audio = audioldm_pipe(**inputs).audios[0]
    assert audio.ndim == 1
    assert len(audio) == 81920
    audio_slice = audio[27780:27790]
    expected_slice = np.array([-0.2131, -0.0873, -0.0124, -0.0189, 0.0569, 
        0.1373, 0.1883, 0.2886, 0.3297, 0.2212])
    max_diff = np.abs(expected_slice - audio_slice).max()
    assert max_diff < 0.03
