def test_embedding_bag(self):
    pre_embed_dim = 1024
    post_embed_dim = 32
    inp = torch.randint(0, pre_embed_dim, (128, 16), device='cuda')
    weight = torch.randn(pre_embed_dim, post_embed_dim, device='cuda',
        dtype=self.dtype)
    output = F.embedding_bag(inp, weight, offsets=None, max_norm=None,
        norm_type=2, scale_grad_by_freq=False, mode='mean', sparse=False)
