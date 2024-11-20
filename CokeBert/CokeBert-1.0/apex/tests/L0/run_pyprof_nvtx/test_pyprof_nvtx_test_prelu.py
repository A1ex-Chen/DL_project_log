def test_prelu(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    weight = torch.randn(1, device='cuda', dtype=self.dtype)
    output = F.prelu(inp, weight)
