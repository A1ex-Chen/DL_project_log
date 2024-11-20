def test_softmin(self):
    inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
    output = F.softmin(inp, dim=1, _stacklevel=3, dtype=self.dtype)
