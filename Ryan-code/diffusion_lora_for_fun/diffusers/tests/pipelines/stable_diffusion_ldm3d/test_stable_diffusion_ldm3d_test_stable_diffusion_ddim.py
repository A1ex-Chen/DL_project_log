def test_stable_diffusion_ddim(self):
    device = 'cpu'
    components = self.get_dummy_components()
    ldm3d_pipe = StableDiffusionLDM3DPipeline(**components)
    ldm3d_pipe = ldm3d_pipe.to(torch_device)
    ldm3d_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = ldm3d_pipe(**inputs)
    rgb, depth = output.rgb, output.depth
    image_slice_rgb = rgb[0, -3:, -3:, -1]
    image_slice_depth = depth[0, -3:, -1]
    assert rgb.shape == (1, 64, 64, 3)
    assert depth.shape == (1, 64, 64)
    expected_slice_rgb = np.array([0.37338176, 0.70247, 0.74203193, 
        0.51643604, 0.58256793, 0.60932136, 0.4181095, 0.48355877, 0.46535262])
    expected_slice_depth = np.array([103.46727, 85.812004, 87.849236])
    assert np.abs(image_slice_rgb.flatten() - expected_slice_rgb).max() < 0.01
    assert np.abs(image_slice_depth.flatten() - expected_slice_depth).max(
        ) < 0.01
