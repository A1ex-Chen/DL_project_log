def test_single_file_components_with_original_config(self, pipe=None,
    single_file_pipe=None):
    pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id,
        safety_checker=None)
    upcast_attention = pipe.unet.config.upcast_attention
    single_file_pipe = (single_file_pipe or self.pipeline_class.
        from_single_file(self.ckpt_path, original_config=self.
        original_config, safety_checker=None, upcast_attention=
        upcast_attention))
    self._compare_component_configs(pipe, single_file_pipe)
