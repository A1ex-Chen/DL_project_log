def test_dropout(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    output = F.dropout(inp, p=0.5, training=True, inplace=False)
