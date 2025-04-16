def test_log_softmax(self):
    inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
    output = F.log_softmax(inp, dim=-1, _stacklevel=3)
