def test_output_is_half(self):
    out_tensor = self.fp16_model(self.in_tensor)
    assert out_tensor.dtype == torch.half
