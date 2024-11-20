def test_softmax(self):
    inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
    output = F.softmax(inp, dim=1, _stacklevel=3, dtype=self.dtype)
