def deepview_input_provider(batch_size=64):
    vocab_size = 32000
    src_len = 25
    tgt_len = 25
    device = torch.device('cuda')
    src = torch.randint(low=0, high=vocab_size, size=(src_len, batch_size),
        dtype=torch.int64, device=device)
    tgt = torch.randint(low=0, high=vocab_size, size=(tgt_len, batch_size),
        dtype=torch.int64, device=device)
    src_len_tensor = torch.tensor([src_len] * batch_size, dtype=torch.int64,
        device=device)
    tgt_len_tensor = torch.tensor([tgt_len] * batch_size, dtype=torch.int64,
        device=device)
    return src, src_len_tensor, tgt, tgt_len_tensor
