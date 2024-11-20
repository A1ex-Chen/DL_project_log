def test_interpolate(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    output = F.interpolate(inp, size=None, scale_factor=2, mode='nearest',
        align_corners=None)
