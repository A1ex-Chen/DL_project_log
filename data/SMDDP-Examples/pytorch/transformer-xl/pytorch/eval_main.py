def main():
    args = parse_args()
    if args.affinity != 'disabled':
        nproc_per_node = torch.cuda.device_count()
        affinity = utils.gpu_affinity.set_affinity(args.local_rank,
            nproc_per_node, args.affinity)
        print(f'{args.local_rank}: thread affinity: {affinity}')
    if args.type == 'pytorch':
        from mem_transformer import MemTransformerLM
    else:
        from inference.mem_transformer_jit import MemTransformerLM
    torch.cuda.set_device(args.local_rank)
    l2_promote()
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.distributed.init_distributed(args.cuda)
    with utils.distributed.sync_workers() as rank:
        if rank == 0:
            create_exp_dir(args.work_dir, debug=args.debug)
    if args.log_all_ranks:
        log_file = f'eval_log_rank_{utils.distributed.get_rank()}.log'
    else:
        log_file = f'eval_log.log'
    dllog_file = args.dllog_file
    log_file = os.path.join(args.work_dir, log_file)
    dllog_file = os.path.join(args.work_dir, dllog_file)
    if args.debug:
        log_file = os.devnull
        dllog_file = os.devnull
    utils.exp_utils.setup_logging(log_all_ranks=args.log_all_ranks,
        filename=log_file, filemode='a')
    utils.exp_utils.setup_dllogger(enabled=True, filename=dllog_file)
    if args.profile:
        try:
            pyprof.init(enable_function_stack=True)
        except NameError:
            warnings.warn('Called pyprof.init() but pyprof is not available')
    logging.info(args)
    dllogger.log(step='PARAMETER', data=vars(args))
    if not args.no_env:
        log_env_info()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.model:
        model_path = args.model
    elif args.work_dir:
        model_path = os.path.join(args.work_dir, 'checkpoint_best.pt')
    else:
        raise RuntimeError(
            'Specify path to checkpoint using --model or --work_dir')
    if not args.manual_config:
        checkpoint = load_checkpoint(model_path)
        vocab_type = checkpoint['args'].vocab
    else:
        checkpoint = None
        vocab_type = args.manual_vocab
    if args.manual:
        vocab = checkpoint['vocab']
        if hasattr(vocab, 'sym2idx') and not hasattr(vocab, 'unk_idx'):
            vocab.unk_idx = vocab.sym2idx['<unk>']
        text = ' '.join(args.manual)
        tokenized = tokenize_raw(text)
        symbols = vocab.tokenize(tokenized, add_eos=True)
        tensor = vocab.convert_to_tensor(symbols)
        iter = data_utils.LMOrderedIterator(tensor, bsz=args.batch_size,
            bptt=args.tgt_len, device=device, ext_len=args.ext_len, warmup=
            False)
    else:
        corpus = get_lm_corpus(args.data, args.dataset, vocab_type)
        if args.split == 'valid' or args.split == 'test':
            iter = corpus.get_iterator(args.split, args.batch_size, args.
                tgt_len, device=device, mem_len=args.mem_len, ext_len=args.
                ext_len)
        else:
            raise RuntimeError('Unknown split')
    if args.fp16:
        dtype = torch.float16
        math_str = 'fp16'
    else:
        dtype = torch.float32
        math_str = 'fp32'
    if args.load_torchscript:
        model = torch.jit.load(args.load_torchscript)
    elif not args.manual_config:
        checkpoint['model_config']['tgt_len'] = args.tgt_len
        checkpoint['model_config']['ext_len'] = args.ext_len
        checkpoint['model_config']['mem_len'] = args.mem_len
        checkpoint['model_config']['clamp_len'] = args.clamp_len
        checkpoint['model_config']['same_length'] = args.same_length
        checkpoint['model_config']['dtype'] = dtype
        model = MemTransformerLM(**checkpoint['model_config'])
        if args.type == 'pytorch':
            model.load_state_dict(checkpoint['model_state'])
        elif args.type == 'torchscript':
            model.load_state_dict(checkpoint['model_state'], strict=False)
    elif args.manual_config:
        args.manual_config['tgt_len'] = args.tgt_len
        args.manual_config['ext_len'] = args.ext_len
        args.manual_config['mem_len'] = args.mem_len
        args.manual_config['clamp_len'] = args.clamp_len
        args.manual_config['same_length'] = args.same_length
        args.manual_config['dtype'] = dtype
        model = MemTransformerLM(**args.manual_config)
    model = model.eval()
    model = model.to(device)
    model = model.to(dtype)
    if args.type == 'torchscript' and not args.manual_config:
        state = checkpoint['model_state']
        tie_projs = checkpoint['model_config']['tie_projs']
        tie_weight = checkpoint['model_config']['tie_weight']
        div_val = checkpoint['model_config']['div_val']
        d_model = checkpoint['model_config']['d_model']
        d_embed = checkpoint['model_config']['d_embed']
        if div_val != 1 or d_model != d_embed:
            for i in range(len(model.word_emb.emb_projs)):
                model.word_emb.emb_projs[i] = state[f'word_emb.emb_projs.{i}'
                    ].to(dtype)
        for i in range(len(model.crit.out_projs)):
            if div_val == 1:
                src = 0
            else:
                src = i
            if model.crit.out_projs[i] is not None:
                if tie_projs[i]:
                    model.crit.out_projs[i] = state[f'word_emb.emb_projs.{src}'
                        ].to(dtype)
                else:
                    model.crit.out_projs[i] = state[f'crit.out_projs.{i}'].to(
                        dtype)
        for i in range(len(model.crit.out_layers_biases)):
            model.crit.out_layers_biases[i] = state[
                f'crit.out_layers_biases.{i}'].to(dtype)
        if tie_weight:
            for i in range(len(model.crit.out_layers_weights)):
                model.crit.out_layers_weights[i] = state[
                    f'word_emb.emb_layers.{i}.weight'].to(dtype)
        else:
            for i in range(len(model.crit.out_layers_weights)):
                model.crit.out_layers_weights[i] = state[
                    f'crit.out_layers_weights.{i}'].to(dtype)
        model = torch.jit.script(model)
    if args.type != 'pytorch':
        compile_model(model, device, args)
    if args.type == 'torchscript' and args.save_torchscript:
        torch.jit.save(model, args.save_torchscript)
    logging.info(
        f'Evaluating with: math {math_str} type {args.type} bsz {args.batch_size} tgt_len {args.tgt_len} ext_len {args.ext_len} mem_len {args.mem_len} clamp_len {args.clamp_len}'
        )
    meters = {}
    warmup = args.mem_len // args.tgt_len + 2
    meters['eval_throughput'] = AverageMeter(warmup=warmup, keep=args.save_data
        )
    meters['eval_latency'] = AverageMeter(warmup=warmup, keep=args.save_data)
    with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
        loss = evaluate(iter, model, meters, args.log_interval, args.
            max_size, args.repeat)
    perplexity = math.exp(loss)
    log_str = format_log(loss, args.split, args)
    summary = {'eval_loss': loss, 'eval_ppl': perplexity}
    logging.info('=' * 100)
    logging.info(log_str)
    logging.info('=' * 100)
    if args.save_data:
        latency_data = np.array(meters['eval_latency'].vals)
        throughput_data = np.array(meters['eval_throughput'].vals)
        precision = 'fp16' if args.fp16 else 'fp32'
        data_fname = f'eval_data_{args.batch_size}_{precision}_{args.type}'
        data_path = os.path.join(args.work_dir, data_fname)
        data = {'args': args, 'throughput': throughput_data, 'latency':
            latency_data}
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f'Throughput Avg: {throughput_data.mean():.2f} tok/s')
        logging.info(f'Latency Avg: {1000.0 * latency_data.mean():.2f} ms')
        for p in args.percentiles:
            logging.info(
                f'Latency {p}%: {1000.0 * np.percentile(latency_data, p):.2f} ms'
                )
        logging.info('=' * 100)
        summary.update({'eval_throughput': throughput_data.mean(),
            'eval_avg_latency': 1000 * latency_data.mean()})
        for p in args.percentiles:
            summary[f'eval_{p}%_latency'] = 1000 * np.percentile(latency_data,
                p)
    dllogger.log(step=tuple(), data=summary)
    passed = benchmark(target_perplexity=args.target_perplexity,
        test_perplexity=perplexity, target_throughput=args.
        target_throughput, test_throughput=meters['eval_throughput'].avg)
    if not passed:
        sys.exit(1)
