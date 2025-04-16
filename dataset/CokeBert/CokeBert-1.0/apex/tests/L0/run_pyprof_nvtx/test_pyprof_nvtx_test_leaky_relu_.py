def test_leaky_relu_(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.leaky_relu_(inp, negative_slope=0.01)
