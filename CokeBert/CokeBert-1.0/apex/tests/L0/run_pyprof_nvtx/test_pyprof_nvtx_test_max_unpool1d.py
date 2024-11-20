def test_max_unpool1d(self):
    inp = torch.randn(1, 16, 32, device='cuda', dtype=self.dtype)
    output, indices = F.max_pool1d(inp, kernel_size=5, stride=2, padding=2,
        return_indices=True, ceil_mode=True)
    output = F.max_unpool1d(output, indices, kernel_size=2, stride=2, padding=2
        )
