def test_single_file_components_with_original_config(self):
    adapter = T2IAdapter.from_pretrained(
        'TencentARC/t2i-adapter-lineart-sdxl-1.0', torch_dtype=torch.float16)
    pipe = self.pipeline_class.from_pretrained(self.repo_id, variant='fp16',
        adapter=adapter, torch_dtype=torch.float16, safety_checker=None)
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        original_config=self.original_config, adapter=adapter)
    self._compare_component_configs(pipe, pipe_single_file)
