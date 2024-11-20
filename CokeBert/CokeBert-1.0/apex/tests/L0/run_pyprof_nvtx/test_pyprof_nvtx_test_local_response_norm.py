def test_local_response_norm(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    output = F.local_response_norm(inp, 2, alpha=0.0001, beta=0.75, k=1.0)
