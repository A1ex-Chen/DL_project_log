def deepview_iteration_provider(model):
    args = get_args()
    opt_config = {'optimizer': args.optimizer, 'lr': args.lr}
    scheduler_config = {'warmup_steps': args.warmup_steps, 'remain_steps':
        args.remain_steps, 'decay_interval': args.decay_interval,
        'decay_steps': args.decay_steps, 'decay_factor': args.decay_factor}
    train_loader_len = 437268
    total_train_iters = train_loader_len // args.train_iter_size * args.epochs
    opt_name = opt_config.pop('optimizer')
    optimizer = torch.optim.__dict__[opt_name](model.parameters(), **opt_config
        )
    scheduler = WarmupMultiStepLR(optimizer, total_train_iters, **
        scheduler_config)
    fp_optimizer = Fp32Optimizer(model, args.grad_clip)

    def iteration(src, src_len, tgt, tgt_len):
        loss = model(src, src_len, tgt, tgt_len)
        loss.backward()
        fp_optimizer.step(optimizer, scheduler, update=True)
    return iteration
