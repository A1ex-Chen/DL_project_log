def test_stable_diffusion_negative_prompt(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = PNDMScheduler(skip_prk_steps=True)
    ldm3d_pipe = StableDiffusionLDM3DPipeline(**components)
    ldm3d_pipe = ldm3d_pipe.to(device)
    ldm3d_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    negative_prompt = 'french fries'
    output = ldm3d_pipe(**inputs, negative_prompt=negative_prompt)
    rgb, depth = output.rgb, output.depth
    rgb_slice = rgb[0, -3:, -3:, -1]
    depth_slice = depth[0, -3:, -1]
    assert rgb.shape == (1, 64, 64, 3)
    assert depth.shape == (1, 64, 64)
    expected_slice_rgb = np.array([0.37044, 0.71811503, 0.7223251, 
        0.48603675, 0.5638391, 0.6364948, 0.42833704, 0.4901315, 0.47926217])
    expected_slice_depth = np.array([107.84738, 84.62802, 89.962135])
    assert np.abs(rgb_slice.flatten() - expected_slice_rgb).max() < 0.01
    assert np.abs(depth_slice.flatten() - expected_slice_depth).max() < 0.01
