def test_single_file_components_local_files_only(self, pipe=None,
    single_file_pipe=None):
    pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id,
        safety_checker=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_filename = self.ckpt_path.split('/')[-1]
        local_ckpt_path = download_single_file_checkpoint(self.repo_id,
            ckpt_filename, tmpdir)
        single_file_pipe = (single_file_pipe or self.pipeline_class.
            from_single_file(local_ckpt_path, safety_checker=None,
            local_files_only=True))
    self._compare_component_configs(pipe, single_file_pipe)
