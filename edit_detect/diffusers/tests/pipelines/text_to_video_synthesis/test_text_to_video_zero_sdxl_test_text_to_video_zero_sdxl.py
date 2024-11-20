def test_text_to_video_zero_sdxl(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    inputs = self.get_dummy_inputs(self.generator_device)
    result = pipe(**inputs).images
    first_frame_slice = result[0, -3:, -3:, -1]
    last_frame_slice = result[-1, -3:, -3:, 0]
    expected_slice1 = np.array([0.48, 0.58, 0.53, 0.59, 0.5, 0.44, 0.6, 
        0.65, 0.52])
    expected_slice2 = np.array([0.66, 0.49, 0.4, 0.7, 0.47, 0.51, 0.73, 
        0.65, 0.52])
    assert np.abs(first_frame_slice.flatten() - expected_slice1).max() < 0.01
    assert np.abs(last_frame_slice.flatten() - expected_slice2).max() < 0.01
