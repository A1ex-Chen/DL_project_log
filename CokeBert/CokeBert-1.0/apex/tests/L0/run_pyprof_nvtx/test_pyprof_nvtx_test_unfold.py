def test_unfold(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    kernel_size = 4, 5
    inp_unf_dilated = F.unfold(inp, kernel_size, dilation=2)
    inp_unf_padded = F.unfold(inp, kernel_size, padding=2)
    inp_unf_strided = F.unfold(inp, kernel_size, stride=2)
