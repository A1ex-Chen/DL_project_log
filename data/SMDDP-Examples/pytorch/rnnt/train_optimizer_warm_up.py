def optimizer_warm_up():
    WARMUP_LEN = 8
    feats = torch.ones(WARMUP_LEN, batch_size, rnnt_config['in_feats'],
        dtype=torch.float16, device='cuda')
    feat_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda'
        ) * WARMUP_LEN
    txt = torch.ones(batch_size, WARMUP_LEN, dtype=torch.int64, device='cuda')
    txt_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda'
        ) * WARMUP_LEN
    dict_meta_data = train_preproc.get_packing_meta_data(feats.size(0),
        feat_lens, txt_lens)
    log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens,
        dict_meta_data)
    loss = loss_fn(log_probs, log_prob_lens, txt, txt_lens, dict_meta_data)
    loss /= args.grad_accumulation_steps
    del log_probs, log_prob_lens
    assert not torch.isnan(loss).any(), 'should not have happened'
    if args.dist_lamb:
        optimizer._lazy_init_stage1()
        grad_scaler.scale(loss).backward()
        optimizer._lazy_init_stage2()
    else:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    optimizer.zero_grad()
    if args.dist_lamb:
        optimizer.complete_reductions()
        optimizer.set_global_scale(grad_scaler._get_scale_async())
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
