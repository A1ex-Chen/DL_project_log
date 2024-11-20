def test_avg_pool2d(self):
    inp = torch.randn(1, 3, 224, 224, device='cuda', dtype=self.dtype)
    out = F.avg_pool2d(inp, kernel_size=5, stride=2, padding=2, ceil_mode=
        True, count_include_pad=False)
