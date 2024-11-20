def test_single_file_components(self):
    adapter = T2IAdapter.from_pretrained(
        'TencentARC/t2i-adapter-lineart-sdxl-1.0', torch_dtype=torch.float16)
    pipe = self.pipeline_class.from_pretrained(self.repo_id, variant='fp16',
        adapter=adapter, torch_dtype=torch.float16)
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        safety_checker=None, adapter=adapter)
    super().test_single_file_components(pipe, pipe_single_file)
