def test_linear(self):
    inp = torch.randn(32, 64, 128, device='cuda', dtype=self.dtype)
    weight = torch.randn(256, 128, device='cuda', dtype=self.dtype)
    output = F.linear(inp, weight, bias=None)
