def test_threshold(self):
    inp = torch.randn(1, 8, 32, 32, device='cuda', dtype=self.dtype)
    output = F.threshold(inp, 6, 6, inplace=False)
