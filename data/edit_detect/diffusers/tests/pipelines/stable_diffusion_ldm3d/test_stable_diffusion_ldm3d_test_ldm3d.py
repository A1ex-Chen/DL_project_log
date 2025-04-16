def test_ldm3d(self):
    ldm3d_pipe = StableDiffusionLDM3DPipeline.from_pretrained('Intel/ldm3d'
        ).to(torch_device)
    ldm3d_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    output = ldm3d_pipe(**inputs)
    rgb, depth = output.rgb, output.depth
    expected_rgb_mean = 0.495586
    expected_rgb_std = 0.33795515
    expected_depth_mean = 112.48518
    expected_depth_std = 98.489746
    assert np.abs(expected_rgb_mean - rgb.mean()) < 0.001
    assert np.abs(expected_rgb_std - rgb.std()) < 0.001
    assert np.abs(expected_depth_mean - depth.mean()) < 0.001
    assert np.abs(expected_depth_std - depth.std()) < 0.001
