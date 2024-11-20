def deepview_input_provider(batch_size=2):
    vocab_size = 30522
    src_seq_len = 512
    tgt_seq_len = 512
    device = torch.device('cuda')
    source = torch.randint(low=0, high=vocab_size, size=(batch_size,
        src_seq_len), dtype=torch.int64, device=device)
    target = torch.randint(low=0, high=vocab_size, size=(batch_size,
        tgt_seq_len), dtype=torch.int64, device=device)
    return source, target
