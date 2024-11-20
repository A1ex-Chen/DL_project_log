def test_audioldm(self):
    audioldm_pipe = AudioLDMPipeline.from_pretrained('cvssp/audioldm')
    audioldm_pipe = audioldm_pipe.to(torch_device)
    audioldm_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 25
    audio = audioldm_pipe(**inputs).audios[0]
    assert audio.ndim == 1
    assert len(audio) == 81920
    audio_slice = audio[77230:77240]
    expected_slice = np.array([-0.4884, -0.4607, 0.0023, 0.5007, 0.5896, 
        0.5151, 0.3813, -0.0208, -0.3687, -0.4315])
    max_diff = np.abs(expected_slice - audio_slice).max()
    assert max_diff < 0.01
