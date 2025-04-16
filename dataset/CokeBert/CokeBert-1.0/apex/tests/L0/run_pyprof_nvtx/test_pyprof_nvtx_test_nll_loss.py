def test_nll_loss(self):
    inp = torch.randn(64, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randint(0, 10, (64,), device='cuda', dtype=torch.long)
    output = F.nll_loss(inp, target, weight=None, size_average=None,
        ignore_index=-100, reduce=None, reduction='mean')
