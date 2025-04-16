def test_single_file_components(self, pipe=None, single_file_pipe=None):
    single_file_pipe = (single_file_pipe or self.pipeline_class.
        from_single_file(self.ckpt_path, safety_checker=None))
    pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id,
        safety_checker=None)
    self._compare_component_configs(pipe, single_file_pipe)
