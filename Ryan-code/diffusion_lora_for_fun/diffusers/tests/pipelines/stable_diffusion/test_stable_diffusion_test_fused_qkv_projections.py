def test_fused_qkv_projections(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    original_image_slice = image[0, -3:, -3:, -1]
    sd_pipe.fuse_qkv_projections()
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice_fused = image[0, -3:, -3:, -1]
    sd_pipe.unfuse_qkv_projections()
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice_disabled = image[0, -3:, -3:, -1]
    assert np.allclose(original_image_slice, image_slice_fused, atol=0.01,
        rtol=0.01), "Fusion of QKV projections shouldn't affect the outputs."
    assert np.allclose(image_slice_fused, image_slice_disabled, atol=0.01,
        rtol=0.01
        ), "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
    assert np.allclose(original_image_slice, image_slice_disabled, atol=
        0.01, rtol=0.01
        ), 'Original outputs should match when fused QKV projections are disabled.'
