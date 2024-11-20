def config_lp_optimizer(model, data, args):
    if args.optimizer == 'adam':
        args.wd = 0
        args.wd_pretrained = 0
        args.wd_new = 0
    in_clap = lambda n, p: n.startswith('clap_model')
    named_parameters = list(model.named_parameters())
    optimizer = {}
    scheduler = {}
    text_freeze_parameters = [p for n, p in named_parameters if n.
        startswith('clap_model.transformer') or n in [
        'clap_model.positional_embedding', 'clap_model.text_projection'] or
        n.startswith('clap_model.token_embedding') or n.startswith(
        'clap_model.ln_final')]
    if args.freeze_text:
        logging.info('Freeze Text!!!!')
        for k in text_freeze_parameters:
            k.requires_grad = False
    if not args.lp_freeze:
        exclude = (lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or 
            'bias' in n or 'logit_scale' in n)
        include = lambda n, p: not exclude(n, p)
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n,
            p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and
            p.requires_grad]
        if args.train_data is None:
            optimizer = None
            scheduler = None
        else:
            total_steps = data['train'].dataloader.num_batches * args.epochs
            if args.split_opt:
                for x in ['lr', 'beta1', 'beta2', 'eps', 'wd']:
                    for y in ['_new', '_pretrained']:
                        if getattr(args, x + y) is None:
                            setattr(args, x + y, getattr(args, x))
                gain_or_bias_pretrained_params = [p for n, p in
                    named_parameters if (exclude(n, p) and p.requires_grad) and
                    is_pretrained_params(n)]
                rest_pretrained_params = [p for n, p in named_parameters if
                    (include(n, p) and p.requires_grad) and
                    is_pretrained_params(n)]
                gain_or_bias_new_params = [p for n, p in named_parameters if
                    (exclude(n, p) and p.requires_grad) and not
                    is_pretrained_params(n)]
                rest_new_params = [p for n, p in named_parameters if (
                    include(n, p) and p.requires_grad) and not
                    is_pretrained_params(n)]
                pretrained_params_optimizer = get_optimizer([{'params':
                    gain_or_bias_pretrained_params, 'weight_decay': 0.0}, {
                    'params': rest_pretrained_params, 'weight_decay': args.
                    wd_pretrained}], lr=args.lr_pretrained, betas=(args.
                    beta1_pretrained, args.beta2_pretrained), eps=args.
                    eps_pretrained, momentum=args.momentum_pretrained,
                    optimizer_name=args.optimizer)
                pretrained_params_scheduler = cosine_lr(
                    pretrained_params_optimizer, args.lr_pretrained, args.
                    warmup, total_steps)
                new_params_optimizer = get_optimizer([{'params':
                    gain_or_bias_new_params, 'weight_decay': 0.0}, {
                    'params': rest_new_params, 'weight_decay': args.wd_new}
                    ], lr=args.lr_new, betas=(args.beta1_new, args.
                    beta2_new), eps=args.eps_new, momentum=args.
                    momentum_new, optimizer_name=args.optimizer)
                new_params_scheduler = cosine_lr(new_params_optimizer, args
                    .lr_new, args.warmup, total_steps)
                optimizer['text'] = pretrained_params_optimizer
                optimizer['audio'] = new_params_optimizer
                scheduler['text'] = pretrained_params_scheduler
                scheduler['audio'] = new_params_scheduler
                if args.horovod:
                    pretrained_params_optimizer = hvd.DistributedOptimizer(
                        pretrained_params_optimizer, named_parameters=model
                        .named_parameters())
                    new_params_optimizer = hvd.DistributedOptimizer(
                        new_params_optimizer, named_parameters=model.
                        named_parameters())
                    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                    hvd.broadcast_optimizer_state(pretrained_params_optimizer,
                        root_rank=0)
                    hvd.broadcast_optimizer_state(new_params_optimizer,
                        root_rank=0)
            else:
                optimizer['clap'] = get_optimizer([{'params':
                    gain_or_bias_params, 'weight_decay': 0.0}, {'params':
                    rest_params, 'weight_decay': args.wd}], lr=args.lr,
                    betas=(args.beta1, args.beta2), eps=args.eps, momentum=
                    args.momentum, optimizer_name=args.optimizer)
                scheduler['clap'] = cosine_lr(optimizer['clap'], args.lr,
                    args.warmup, total_steps)
                if args.horovod:
                    optimizer['clap'] = hvd.DistributedOptimizer(optimizer[
                        'clap'], named_parameters=model.named_parameters())
                    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                    hvd.broadcast_optimizer_state(optimizer['clap'],
                        root_rank=0)
    else:
        lp_params = [p for n, p in named_parameters if not in_clap(n, p) and
            p.requires_grad]
        lp_optim = get_optimizer(lp_params, lr=args.lp_lr, betas=(args.
            beta1, args.beta2), eps=args.eps, momentum=0.9, optimizer_name=
            args.optimizer)
        optimizer['lp'] = lp_optim
    return optimizer, scheduler, text_freeze_parameters
