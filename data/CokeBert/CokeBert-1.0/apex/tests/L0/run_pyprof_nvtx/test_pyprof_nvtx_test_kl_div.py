def test_kl_div(self):
    inp = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    output = F.kl_div(inp, target, size_average=None, reduce=None,
        reduction='batchmean')
