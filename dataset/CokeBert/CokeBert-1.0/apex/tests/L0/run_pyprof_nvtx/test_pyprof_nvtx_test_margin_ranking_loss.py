def test_margin_ranking_loss(self):
    inp1 = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    inp2 = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = (torch.randint(0, 1, (128,), device='cuda') - 1).type_as(inp1)
    output = F.margin_ranking_loss(inp1, inp2, target, margin=0,
        size_average=None, reduce=None, reduction='mean')
