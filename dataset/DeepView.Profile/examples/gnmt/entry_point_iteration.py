def iteration(src, src_len, tgt, tgt_len):
    loss = model(src, src_len, tgt, tgt_len)
    loss.backward()
    fp_optimizer.step(optimizer, scheduler, update=True)
