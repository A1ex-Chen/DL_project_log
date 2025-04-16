def test_hinge_embedding_loss(self):
    inp = torch.randn(128, 32, device='cuda', dtype=self.dtype)
    target = torch.randint(0, 1, (32,), device='cuda') - 1
    output = F.hinge_embedding_loss(inp, target, margin=1.0, size_average=
        None, reduce=None, reduction='mean')
