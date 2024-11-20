def test_ldm3d_stable_diffusion(self):
    ldm3d_pipe = StableDiffusionLDM3DPipeline.from_pretrained('Intel/ldm3d')
    ldm3d_pipe = ldm3d_pipe.to(torch_device)
    ldm3d_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    output = ldm3d_pipe(**inputs)
    rgb, depth = output.rgb, output.depth
    rgb_slice = rgb[0, -3:, -3:, -1].flatten()
    depth_slice = rgb[0, -3:, -1].flatten()
    assert rgb.shape == (1, 512, 512, 3)
    assert depth.shape == (1, 512, 512)
    expected_slice_rgb = np.array([0.53805465, 0.56707305, 0.5486515, 
        0.57012236, 0.5814511, 0.56253487, 0.54843014, 0.55092263, 0.6459706])
    expected_slice_depth = np.array([0.9263781, 0.6678672, 0.5486515, 
        0.92202145, 0.67831135, 0.56253487, 0.9241694, 0.7551478, 0.6459706])
    assert np.abs(rgb_slice - expected_slice_rgb).max() < 0.003
    assert np.abs(depth_slice - expected_slice_depth).max() < 0.003
