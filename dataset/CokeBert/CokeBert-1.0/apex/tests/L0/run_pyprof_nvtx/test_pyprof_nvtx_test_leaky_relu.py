def test_leaky_relu(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.leaky_relu(inp, negative_slope=0.01, inplace=False)
