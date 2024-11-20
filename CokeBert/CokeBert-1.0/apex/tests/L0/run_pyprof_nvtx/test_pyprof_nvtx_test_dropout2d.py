def test_dropout2d(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    output = F.dropout2d(inp, p=0.5, training=True, inplace=False)
