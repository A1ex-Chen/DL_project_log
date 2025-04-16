def test_sigmoid(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = torch.sigmoid(inp)
