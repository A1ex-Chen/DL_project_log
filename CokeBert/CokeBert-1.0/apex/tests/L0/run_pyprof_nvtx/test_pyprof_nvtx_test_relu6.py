def test_relu6(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.relu6(inp, inplace=False)
