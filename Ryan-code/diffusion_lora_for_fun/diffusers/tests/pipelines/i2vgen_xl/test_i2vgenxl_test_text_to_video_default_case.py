def test_text_to_video_default_case(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['output_type'] = 'np'
    frames = pipe(**inputs).frames
    image_slice = frames[0][0][-3:, -3:, -1]
    assert frames[0][0].shape == (32, 32, 3)
    expected_slice = np.array([0.5146, 0.6525, 0.6032, 0.5204, 0.5675, 
        0.4125, 0.3016, 0.5172, 0.4095])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
