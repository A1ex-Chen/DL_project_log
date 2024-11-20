def test_ldm3d_v2(self):
    ldm3d_pipe = StableDiffusionLDM3DPipeline.from_pretrained('Intel/ldm3d-4c'
        ).to(torch_device)
    ldm3d_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    output = ldm3d_pipe(**inputs)
    rgb, depth = output.rgb, output.depth
    expected_rgb_mean = 0.4194127
    expected_rgb_std = 0.35375586
    expected_depth_mean = 0.5638502
    expected_depth_std = 0.34686103
    assert rgb.shape == (1, 512, 512, 3)
    assert depth.shape == (1, 512, 512, 1)
    assert np.abs(expected_rgb_mean - rgb.mean()) < 0.001
    assert np.abs(expected_rgb_std - rgb.std()) < 0.001
    assert np.abs(expected_depth_mean - depth.mean()) < 0.001
    assert np.abs(expected_depth_std - depth.std()) < 0.001
