def test_fused_qkv_projections(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['return_dict'] = False
    image = pipe(**inputs)[0]
    original_image_slice = image[0, -3:, -3:, -1]
    pipe.fuse_qkv_projections()
    inputs = self.get_dummy_inputs(device)
    inputs['return_dict'] = False
    image_fused = pipe(**inputs)[0]
    image_slice_fused = image_fused[0, -3:, -3:, -1]
    pipe.unfuse_qkv_projections()
    inputs = self.get_dummy_inputs(device)
    inputs['return_dict'] = False
    image_disabled = pipe(**inputs)[0]
    image_slice_disabled = image_disabled[0, -3:, -3:, -1]
    assert np.allclose(original_image_slice, image_slice_fused, atol=0.01,
        rtol=0.01), "Fusion of QKV projections shouldn't affect the outputs."
    assert np.allclose(image_slice_fused, image_slice_disabled, atol=0.01,
        rtol=0.01
        ), "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
    assert np.allclose(original_image_slice, image_slice_disabled, atol=
        0.01, rtol=0.01
        ), 'Original outputs should match when fused QKV projections are disabled.'
