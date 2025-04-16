def test_single_file_arguments(self):
    model_default = self.model_class.from_single_file(self.ckpt_path)
    assert model_default.config.upcast_attention is False
    assert model_default.dtype == torch.float32
    torch_dtype = torch.float16
    upcast_attention = True
    model = self.model_class.from_single_file(self.ckpt_path,
        upcast_attention=upcast_attention, torch_dtype=torch_dtype)
    assert model.config.upcast_attention == upcast_attention
    assert model.dtype == torch_dtype
