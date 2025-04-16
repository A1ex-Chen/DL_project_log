def test_multilabel_margin_loss(self):
    inp = torch.randn(1024, device='cuda', dtype=self.dtype, requires_grad=True
        )
    target = torch.randint(0, 10, (1024,), dtype=torch.long, device='cuda')
    output = F.multilabel_margin_loss(inp, target, size_average=None,
        reduce=None, reduction='mean')
