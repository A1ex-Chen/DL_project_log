def deepview_iteration_provider(transformer):
    opt = model_config()
    optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad,
        transformer.parameters()), betas=(0.9, 0.98), eps=1e-09), opt.
        d_model, opt.n_warmup_steps)

    def iteration(src_seq, src_pos, tgt_seq, tgt_pos, gold):
        optimizer.zero_grad()
        loss = transformer(src_seq, src_pos, tgt_seq, tgt_pos, gold)
        loss.backward()
        optimizer.step_and_update_lr()
    return iteration
