def test_raises_warning_for_mask_feature(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs.update({'mask_feature': True})
    with self.assertWarns(FutureWarning) as warning_ctx:
        _ = pipe(**inputs).images
    assert 'mask_feature' in str(warning_ctx.warning)
