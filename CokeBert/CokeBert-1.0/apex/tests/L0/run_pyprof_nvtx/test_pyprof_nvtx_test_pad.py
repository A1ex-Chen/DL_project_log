def test_pad(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    pad = 3, 3
    output = F.pad(inp, pad, mode='constant', value=0)
