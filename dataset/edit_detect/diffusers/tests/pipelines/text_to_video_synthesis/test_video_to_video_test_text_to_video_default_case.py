def test_text_to_video_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = VideoToVideoSDPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['output_type'] = 'np'
    frames = sd_pipe(**inputs).frames
    image_slice = frames[0][0][-3:, -3:, -1]
    assert frames[0][0].shape == (32, 32, 3)
    expected_slice = np.array([0.6391, 0.535, 0.5202, 0.5521, 0.5453, 
        0.5393, 0.6652, 0.527, 0.5185])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
