def test_single_file_components_with_original_config_local_files_only(self,
    pipe=None, single_file_pipe=None):
    pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id,
        safety_checker=None)
    upcast_attention = pipe.unet.config.upcast_attention
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_filename = self.ckpt_path.split('/')[-1]
        local_ckpt_path = download_single_file_checkpoint(self.repo_id,
            ckpt_filename, tmpdir)
        local_original_config = download_original_config(self.
            original_config, tmpdir)
        single_file_pipe = (single_file_pipe or self.pipeline_class.
            from_single_file(local_ckpt_path, original_config=
            local_original_config, upcast_attention=upcast_attention,
            safety_checker=None, local_files_only=True))
    self._compare_component_configs(pipe, single_file_pipe)
