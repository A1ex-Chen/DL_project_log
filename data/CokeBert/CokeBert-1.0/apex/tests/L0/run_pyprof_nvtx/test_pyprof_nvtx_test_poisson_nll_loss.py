def test_poisson_nll_loss(self):
    inp = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=False)
    output = F.poisson_nll_loss(inp, target, log_input=True, full=False,
        size_average=None, eps=1e-08, reduce=None, reduction='mean')
