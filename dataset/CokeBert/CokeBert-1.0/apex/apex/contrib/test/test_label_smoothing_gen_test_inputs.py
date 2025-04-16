def gen_test_inputs(self, N, T, H, smoothing, padding_idx):
    logits = torch.randn((N * T, H), dtype=torch.half, device='cuda',
        requires_grad=True)
    labels = torch.randint(0, H, [N * T], device='cuda')
    for i in random.sample(range(N * T), N * T // 6):
        labels[i] = padding_idx
    half_to_float = logits.dtype == torch.half
    return logits, labels, half_to_float
