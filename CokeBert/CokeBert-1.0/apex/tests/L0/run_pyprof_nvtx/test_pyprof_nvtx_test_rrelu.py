def test_rrelu(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.rrelu(inp, lower=1.0 / 8, upper=1.0 / 3, training=False,
        inplace=False)
