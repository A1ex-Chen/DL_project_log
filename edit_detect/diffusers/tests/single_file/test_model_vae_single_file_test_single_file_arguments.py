def test_single_file_arguments(self):
    model_default = self.model_class.from_single_file(self.ckpt_path,
        config=self.repo_id)
    assert model_default.config.scaling_factor == 0.18215
    assert model_default.config.sample_size == 256
    assert model_default.dtype == torch.float32
    scaling_factor = 2.0
    sample_size = 512
    torch_dtype = torch.float16
    model = self.model_class.from_single_file(self.ckpt_path, config=self.
        repo_id, sample_size=sample_size, scaling_factor=scaling_factor,
        torch_dtype=torch_dtype)
    assert model.config.scaling_factor == scaling_factor
    assert model.config.sample_size == sample_size
    assert model.dtype == torch_dtype
