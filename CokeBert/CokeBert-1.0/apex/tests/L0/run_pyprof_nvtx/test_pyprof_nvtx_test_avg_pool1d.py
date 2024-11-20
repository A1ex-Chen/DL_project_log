def test_avg_pool1d(self):
    inp = torch.randn(1, 1, 28, device='cuda', dtype=self.dtype)
    out = F.avg_pool1d(inp, kernel_size=5, stride=2, padding=2, ceil_mode=
        True, count_include_pad=False)
