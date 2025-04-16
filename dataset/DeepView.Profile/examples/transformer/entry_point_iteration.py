def iteration(src_seq, src_pos, tgt_seq, tgt_pos, gold):
    optimizer.zero_grad()
    loss = transformer(src_seq, src_pos, tgt_seq, tgt_pos, gold)
    loss.backward()
    optimizer.step_and_update_lr()
