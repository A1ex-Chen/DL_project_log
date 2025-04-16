def test_max_unpool3d(self):
    inp = torch.randn(1, 16, 8, 32, 32, device='cuda', dtype=self.dtype)
    output, indices = F.max_pool3d(inp, kernel_size=5, stride=2, padding=2,
        return_indices=True, ceil_mode=True)
    output = F.max_unpool3d(output, indices, kernel_size=2, stride=2, padding=2
        )
