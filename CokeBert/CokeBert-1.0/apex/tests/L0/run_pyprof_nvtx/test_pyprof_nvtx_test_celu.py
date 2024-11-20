def test_celu(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.celu(inp, alpha=1.0, inplace=False)
