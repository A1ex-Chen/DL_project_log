def test_gumbel_softmax(self):
    inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
    output = F.gumbel_softmax(inp, tau=1, hard=False, eps=1e-10, dim=-1)
