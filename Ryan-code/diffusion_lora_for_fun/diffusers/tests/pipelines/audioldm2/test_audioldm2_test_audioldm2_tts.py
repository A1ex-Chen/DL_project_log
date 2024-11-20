def test_audioldm2_tts(self):
    audioldm_tts_pipe = AudioLDM2Pipeline.from_pretrained(
        'anhnct/audioldm2_gigaspeech')
    audioldm_tts_pipe = audioldm_tts_pipe.to(torch_device)
    audioldm_tts_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs_tts(torch_device)
    audio = audioldm_tts_pipe(**inputs).audios[0]
    assert audio.ndim == 1
    assert len(audio) == 81952
    audio_slice = audio[8825:8835]
    expected_slice = np.array([-0.1829, -0.1461, 0.0759, -0.1493, -0.1396, 
        0.5783, 0.3001, -0.3038, -0.0639, -0.2244])
    max_diff = np.abs(expected_slice - audio_slice).max()
    assert max_diff < 0.001
