def test_save_load_local(self, expected_max_difference=0.0005):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    output = pipe(**inputs)[0]
    logger = logging.get_logger('diffusers.pipelines.pipeline_utils')
    logger.setLevel(diffusers.logging.INFO)
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir, safe_serialization=False)
        with CaptureLogger(logger) as cap_logger:
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
        for component in pipe_loaded.components.values():
            if hasattr(component, 'set_default_attn_processor'):
                component.set_default_attn_processor()
        for name in pipe_loaded.components.keys():
            if name not in pipe_loaded._optional_components:
                assert name in str(cap_logger)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    output_loaded = pipe_loaded(**inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
    self.assertLess(max_diff, expected_max_difference)
