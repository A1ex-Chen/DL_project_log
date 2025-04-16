def test_dropout3d(self):
    inp = torch.randn(16, 8, 32, 64, 64, device='cuda', dtype=self.dtype)
    output = F.dropout3d(inp, p=0.5, training=True, inplace=False)
