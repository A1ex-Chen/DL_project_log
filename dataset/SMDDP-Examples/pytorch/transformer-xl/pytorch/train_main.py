def main():
    args = parse_args()
    if args.affinity != 'disabled':
        nproc_per_node = torch.cuda.device_count()
        affinity = utils.gpu_affinity.set_affinity(args.local_rank,
            nproc_per_node, args.affinity)
        print(f'{args.local_rank}: thread affinity: {affinity}')
    torch.cuda.set_device(args.local_rank)
    l2_promote()
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.work_dir = utils.exp_utils.build_work_dir_name(args.work_dir, args
        .dataset, args.append_dataset, args.append_time)
    with utils.distributed.sync_workers() as rank:
        if rank == 0:
            create_exp_dir(args.work_dir, scripts_to_save=['train.py',
                'mem_transformer.py'], debug=args.debug)
    if args.log_all_ranks:
        log_file = f'train_log_rank_{utils.distributed.get_rank()}.log'
    else:
        log_file = args.txtlog_file
    dllog_file = args.dllog_file
    log_file = os.path.join(args.work_dir, log_file)
    dllog_file = os.path.join(args.work_dir, dllog_file)
    if args.debug:
        log_file = os.devnull
        dllog_file = os.devnull
    utils.exp_utils.setup_logging(log_all_ranks=args.log_all_ranks,
        filename=log_file)
    utils.exp_utils.setup_dllogger(enabled=True, filename=dllog_file)
    if args.local_batch_size is not None:
        world_size = utils.distributed.get_world_size()
        args.batch_size = world_size * args.local_batch_size
        logging.info(
            f'--local_batch_size was set, adjusting global batch size to {args.batch_size} (local_batch_size * world_size)'
            )
    if args.profile:
        try:
            pyprof.init(enable_function_stack=True)
        except NameError:
            warnings.warn('Called pyprof.init() but pyprof is not available')
    logging.info(args)
    dllogger.log(step='PARAMETER', data=vars(args))
    logging.info(f'world size: {utils.distributed.get_world_size()}')
    if not args.no_env:
        log_env_info()
    register_ignoring_timeout_handler()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    corpus = get_lm_corpus(args.data, args.dataset, args.vocab)
    ntokens = len(corpus.vocab)
    vocab = corpus.vocab
    args.n_token = ntokens
    if args.mem_len == 0:
        eval_mem_len = 0
    else:
        eval_mem_len = args.mem_len + args.tgt_len - args.eval_tgt_len
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
        device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.
        eval_tgt_len, device=device, mem_len=eval_mem_len, ext_len=args.ext_len
        )
    te_iter = corpus.get_iterator('test', args.eval_batch_size, args.
        eval_tgt_len, device=device, mem_len=eval_mem_len, ext_len=args.ext_len
        )
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [19997, 39997, 199997]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [59997, 99997, 639997]
            tie_projs += [False] * len(cutoffs)
    model_config = {'n_token': ntokens, 'n_layer': args.n_layer, 'n_head':
        args.n_head, 'd_model': args.d_model, 'd_head': args.d_head,
        'd_inner': args.d_inner, 'dropout': args.dropout, 'dropatt': args.
        dropatt, 'dtype': None, 'tie_weight': args.tied, 'd_embed': args.
        d_embed, 'div_val': args.div_val, 'tie_projs': tie_projs,
        'pre_lnorm': args.pre_lnorm, 'tgt_len': args.tgt_len, 'ext_len':
        args.ext_len, 'mem_len': args.mem_len, 'cutoffs': cutoffs,
        'same_length': args.same_length, 'attn_type': args.attn_type,
        'clamp_len': args.clamp_len, 'sample_softmax': args.sample_softmax}
    model = MemTransformerLM(**model_config)
    model.apply(functools.partial(weights_init, args=args))
    model.word_emb.apply(functools.partial(weights_init, args=args))
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()]
        )
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=
                args.mom)
            optimizer_sparse = None
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr, weight_decay=
                args.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay)
            optimizer_sparse = None
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        optimizer_sparse = None
    elif args.optim.lower() == 'lamb':
        optimizer = lamb.Lamb(model.parameters(), lr=args.lr, weight_decay=
            args.weight_decay)
        optimizer_sparse = None
    elif args.optim.lower() == 'jitlamb':
        optimizer = lamb.JITLamb(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        optimizer_sparse = None
    model = model.to(device)
    scaler = None
    if args.fp16:
        if args.amp == 'pytorch':
            scaler = torch.cuda.amp.GradScaler()
        elif args.amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level=
                args.apex_amp_opt_level)
    if args.multi_gpu == 'ddp' and dist.is_initialized():
        para_model = DistributedDataParallel(model, device_ids=[args.
            local_rank], output_device=args.local_rank, broadcast_buffers=
            False, find_unused_parameters=False)
    elif args.multi_gpu == 'dp':
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.
                batch_chunk, model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model
    if args.scheduler == 'cosine':
        if args.max_step_scheduler:
            max_step = args.max_step_scheduler
        else:
            max_step = args.max_step
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
            max_step - args.warmup_step, eta_min=args.eta_min)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_sparse, max_step - args.warmup_step, eta_min=args
                .eta_min)
        else:
            scheduler_sparse = None
    elif args.scheduler == 'inv_sqrt':

        def lr_lambda(step):
            if step == 0 and args.warmup_step == 0:
                return 1.0
            else:
                return (1.0 / step ** 0.5 if step > args.warmup_step else 
                    step / args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.LambdaLR(optimizer_sparse,
                lr_lambda=lr_lambda)
        else:
            scheduler_sparse = None
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=
            args.decay_rate, patience=args.patience, min_lr=args.lr_min)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_sparse, factor=args.decay_rate, patience=args.
                patience, min_lr=args.lr_min)
        else:
            scheduler_sparse = None
    elif args.scheduler == 'constant':
        pass
    logging.info('=' * 100)
    for k, v in args.__dict__.items():
        logging.info('    - {} : {}'.format(k, v))
    logging.info('=' * 100)
    logging.info('#params = {}'.format(args.n_all_param))
    logging.info('#non emb params = {}'.format(args.n_nonemb_param))
    train_step = 0
    start_epoch = 1
    last_batch = 0
    last_iter = 0
    best_val_loss = None
    if args.restart:
        try:
            checkpoint = load_checkpoint(args.restart)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            if args.fp16:
                if args.amp == 'pytorch':
                    scaler.load_state_dict(checkpoint['amp_state'])
                elif args.amp == 'apex':
                    amp.load_state_dict(checkpoint['amp_state'])
            train_step = checkpoint['train_step']
            start_epoch = checkpoint['epoch']
            last_batch = checkpoint['batch']
            last_iter = checkpoint['last_iter']
            best_val_loss = checkpoint['best_val_loss']
            if train_step >= args.max_step:
                logging.info(
                    f'Loaded checkpoint after {train_step} steps, but this run was scheduled for a total of {args.max_step} steps, exiting'
                    )
                sys.exit(1)
            model.apply(functools.partial(update_dropout, args=args))
            model.apply(functools.partial(update_dropatt, args=args))
        except FileNotFoundError:
            logging.info(
                f'Could not load checkpoint from {args.restart}, starting training from random init'
                )
    meters = {}
    warmup = args.mem_len // args.tgt_len + 2
    meters['train_throughput'] = AverageMeter(warmup=warmup)
    start_time = time.time()
    with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
        with TimeoutHandler() as timeout_handler:
            try:
                for epoch in itertools.count(start=start_epoch):
                    if args.roll:
                        tr_iter.roll(seed=args.seed + epoch)
                    train_step, best_val_loss = train(tr_iter, va_iter,
                        model, para_model, model_config, optimizer,
                        optimizer_sparse, scheduler, scheduler_sparse,
                        scaler, vocab, epoch, last_batch, last_iter,
                        train_step, best_val_loss, meters, timeout_handler,
                        device, args)
                    last_batch = 0
                    last_iter = 0
                    if train_step == args.max_step:
                        logging.info('-' * 100)
                        logging.info('End of training')
                        break
            except KeyboardInterrupt:
                logging.info('-' * 100)
                logging.info('Exiting from training early')
    elapsed = time.time() - start_time
    summary = {}
    test_path = os.path.join(args.work_dir, 'checkpoint_best.pt')
    if not args.debug and not args.no_eval and os.path.exists(test_path):
        checkpoint = load_checkpoint(test_path)
        model.load_state_dict(checkpoint['model_state'])
        test_start_time = time.time()
        with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
            test_loss = evaluate(te_iter, model, args)
            test_loss = utils.distributed.all_reduce_item(test_loss, 'mean')
        test_elapsed = time.time() - test_start_time
        logging.info('=' * 100)
        if args.dataset in ['enwik8', 'text8']:
            logging.info(
                '| End of training | test time: {:5.2f}s | test loss {:5.2f} | test bpc {:9.5f}'
                .format(test_elapsed, test_loss, test_loss / math.log(2)))
        else:
            logging.info(
                '| End of training | test time: {:5.2f}s | test loss {:5.2f} | test ppl {:9.3f}'
                .format(test_elapsed, test_loss, math.exp(test_loss)))
        logging.info('=' * 100)
        summary.update({'test_elapsed': test_elapsed, 'test_loss': test_loss})
        if args.dataset in ['enwik8', 'text8']:
            summary['test_bits_per_character'] = test_loss / math.log(2)
        else:
            summary['test_perplexity'] = math.exp(test_loss)
    logging.info(f'Training time: {elapsed / 60:.2f} minutes')
    logging.info(
        f"Training throughput: {meters['train_throughput'].avg:.2f} tok/s")
    if best_val_loss:
        val_perplexity = math.exp(best_val_loss)
    else:
        val_perplexity = None
    summary.update({'train_throughput': meters['train_throughput'].avg,
        'train_elapsed': elapsed / 60, 'valid_loss': best_val_loss,
        'valid_perplexity': val_perplexity})
    dllogger.log(step=tuple(), data=summary)
    passed = benchmark(target_perplexity=args.target_perplexity,
        test_perplexity=val_perplexity, target_throughput=args.
        target_throughput, test_throughput=meters['train_throughput'].avg)
    if not passed:
        sys.exit(1)
