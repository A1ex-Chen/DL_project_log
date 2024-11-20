def test_hardtanh_(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.hardtanh_(inp, min_val=-1.0, max_val=1.0)
