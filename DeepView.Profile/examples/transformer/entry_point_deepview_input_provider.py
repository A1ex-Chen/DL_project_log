def deepview_input_provider(batch_size=64):
    vocab_size = 32000
    src_seq_len = 25
    tgt_seq_len = 25
    device = torch.device('cuda')
    source = torch.randint(low=0, high=vocab_size, size=(batch_size,
        src_seq_len), dtype=torch.int64, device=device)
    target = torch.randint(low=0, high=vocab_size, size=(batch_size,
        tgt_seq_len), dtype=torch.int64, device=device)
    source_pos_row = torch.arange(0, src_seq_len, dtype=torch.int64, device
        =device).unsqueeze_(0)
    target_pos_row = torch.arange(0, tgt_seq_len, dtype=torch.int64, device
        =device).unsqueeze_(0)
    source_pos_list = [source_pos_row for _ in range(batch_size)]
    target_pos_list = [target_pos_row for _ in range(batch_size)]
    src_pos = torch.cat(source_pos_list, 0)
    tgt_pos = torch.cat(target_pos_list, 0)
    gold = target[:, 1:]
    return source, src_pos, target, tgt_pos, gold
