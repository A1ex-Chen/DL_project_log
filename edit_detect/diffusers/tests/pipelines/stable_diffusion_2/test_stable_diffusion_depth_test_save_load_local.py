def test_save_load_local(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    output = pipe(**inputs)[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir)
        pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    output_loaded = pipe_loaded(**inputs)[0]
    max_diff = np.abs(output - output_loaded).max()
    self.assertLess(max_diff, 0.0001)
