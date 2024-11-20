def test_pairwise_distance(self):
    inp1 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
    inp2 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
    output = F.pairwise_distance(inp1, inp2, p=2.0, eps=1e-06, keepdim=False)
