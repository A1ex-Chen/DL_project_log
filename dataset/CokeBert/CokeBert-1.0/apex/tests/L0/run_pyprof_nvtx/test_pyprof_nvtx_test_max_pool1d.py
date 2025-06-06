def test_max_pool1d(self):
    inp = torch.randn(1, 16, 32, device='cuda', dtype=self.dtype)
    out = F.max_pool1d(inp, kernel_size=5, stride=2, padding=2,
        return_indices=True, ceil_mode=True)
