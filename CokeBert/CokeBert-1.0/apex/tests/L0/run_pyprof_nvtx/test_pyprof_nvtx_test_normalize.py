def test_normalize(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    output = F.normalize(inp, p=2, dim=1, eps=1e-12, out=None)
