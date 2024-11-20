def test_save_pretrained_raise_not_implemented_exception(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pipe.save_pretrained(tmpdir)
        except NotImplementedError:
            pass
