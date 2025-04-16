def test_progress_bar(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    inputs = self.get_dummy_inputs(torch_device)
    with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
        _ = pipe(**inputs)
        stderr = stderr.getvalue()
        max_steps = re.search('/(.*?) ', stderr).group(1)
        self.assertTrue(max_steps is not None and len(max_steps) > 0)
        self.assertTrue(f'{max_steps}/{max_steps}' in stderr,
            'Progress bar should be enabled and stopped at the max step')
    pipe.set_progress_bar_config(disable=True)
    with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
        _ = pipe(**inputs)
        self.assertTrue(stderr.getvalue() == '',
            'Progress bar should be disabled')
