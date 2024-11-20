def main():
    args = parse_args()
    logging.configure_logger(args.output_dir, 'RNNT')
    logging.log_start(logging.constants.INIT_START)
    assert torch.cuda.is_available()
    assert args.prediction_frequency is None or args.prediction_frequency % args.log_frequency == 0
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    multi_gpu = True
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        world_size = dist.get_world_size()
        print('local_rank is, ', args.local_rank)
        print('world size is, ', dist.get_world_size())
        print_once(f'Distributed training with {world_size} GPUs\n')
    else:
        world_size = 1
    if args.seed is not None:
        logging.log_event(logging.constants.SEED, value=args.seed)
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)
        sampler_seed = args.seed
        if multi_gpu:
            dali_seed = args.seed + dist.get_rank()
        else:
            dali_seed = args.seed
    init_log(args)
    cfg = config.load(args.model_config)
    config.apply_duration_flags(cfg, args.max_duration)
    assert args.grad_accumulation_steps >= 1
    assert args.batch_size % args.grad_accumulation_steps == 0, f'{args.batch_size} % {args.grad_accumulation_steps} != 0'
    logging.log_event(logging.constants.GRADIENT_ACCUMULATION_STEPS, value=
        args.grad_accumulation_steps)
    batch_size = args.batch_size // args.grad_accumulation_steps
    if args.batch_split_factor != 1:
        assert batch_size % args.batch_split_factor == 0, f'{batch_size} % {args.batch_split_factor} != 0'
        assert args.dist_lamb, 'dist LAMB must be used when batch split is enabled'
    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)
    logging.log_event(logging.constants.SUBMISSION_BENCHMARK, value=logging
        .constants.RNNT)
    logging.log_event(logging.constants.SUBMISSION_ORG, value='NVIDIA')
    logging.log_event(logging.constants.SUBMISSION_DIVISION, value=logging.
        constants.CLOSED)
    logging.log_event(logging.constants.SUBMISSION_STATUS, value=logging.
        constants.ONPREM)
    logging.log_event(logging.constants.SUBMISSION_PLATFORM, value=
        f'{num_nodes}xSUBMISSION_PLATFORM_PLACEHOLDER')
    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)
    rnnt_config = config.rnnt(cfg)
    logging.log_event(logging.constants.MODEL_WEIGHTS_INITIALIZATION_SCALE,
        value=args.weights_init_scale)
    if args.fuse_relu_dropout:
        rnnt_config['fuse_relu_dropout'] = True
    if args.apex_transducer_joint is not None:
        rnnt_config['apex_transducer_joint'] = args.apex_transducer_joint
    if args.weights_init_scale is not None:
        rnnt_config['weights_init_scale'] = args.weights_init_scale
    if args.hidden_hidden_bias_scale is not None:
        rnnt_config['hidden_hidden_bias_scale'] = args.hidden_hidden_bias_scale
    if args.multilayer_lstm:
        rnnt_config['decoupled_rnns'] = False
    if args.apex_mlp:
        rnnt_config['apex_mlp'] = True
    enc_stack_time_factor = rnnt_config['enc_stack_time_factor']
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    model.cuda()
    blank_idx = tokenizer.num_labels
    if args.apex_transducer_loss == None:
        loss_fn = RNNTLoss(blank_idx=blank_idx)
    else:
        if args.apex_transducer_loss == 'fp16':
            assert args.amp_level in [1, 2
                ], 'model in FP32 and loss in FP16 is not a valid use case'
        loss_fn = apexTransducerLoss(blank_idx, args.apex_transducer_loss,
            packed_input=args.apex_transducer_joint == 'pack')
    logging.log_event(logging.constants.EVAL_MAX_PREDICTION_SYMBOLS, value=
        args.max_symbol_per_sample)
    greedy_decoder = RNNTGreedyDecoder(blank_idx=blank_idx, batch_eval_mode
        =args.batch_eval_mode, cg_unroll_factor=args.cg_unroll_factor,
        rnnt_config=rnnt_config, max_symbol_per_sample=args.
        max_symbol_per_sample, amp_level=args.amp_level)
    print_once(f'Model size: {num_weights(model) / 10 ** 6:.1f}M params\n')
    if args.ema > 0:
        ema_model = copy.deepcopy(model).cuda()
    else:
        ema_model = None
    logging.log_event(logging.constants.MODEL_EVAL_EMA_FACTOR, value=args.ema)
    opt_eps = 1e-09
    if args.dist_lamb:
        model.half()
        initial_lrs = args.lr * torch.tensor(1, device='cuda', dtype=torch.
            float)
        kw = {'params': model.parameters(), 'lr': initial_lrs,
            'weight_decay': args.weight_decay, 'eps': opt_eps, 'betas': (
            args.beta1, args.beta2), 'max_grad_norm': args.clip_norm,
            'overlap_reductions': True, 'dwu_group_size': args.
            dwu_group_size, 'use_nvlamb': False, 'dwu_num_blocks': 2,
            'dwu_num_chunks': 2, 'dwu_num_rs_pg': 1, 'dwu_num_ar_pg': 1,
            'dwu_num_ag_pg': 2, 'bias_correction': True}
        optimizer = DistributedFusedLAMB(**kw)
        os.environ['NCCL_SHARP_DISABLE'] = '1'
        grad_scaler = GradScaler(init_scale=512)
        print_once(f'Starting with LRs: {initial_lrs}')
    else:
        grad_scaler = None
        kw = {'params': model.parameters(), 'lr': args.lr, 'max_grad_norm':
            args.clip_norm, 'weight_decay': args.weight_decay}
        initial_lrs = args.lr
        print_once(f'Starting with LRs: {initial_lrs}')
        optimizer = FusedLAMB(betas=(args.beta1, args.beta2), eps=opt_eps, **kw
            )
        model, optimizer = amp.initialize(models=model, optimizers=
            optimizer, opt_level=f'O{args.amp_level}', max_loss_scale=512.0,
            cast_model_outputs=torch.float16 if args.amp_level == 2 else None)
    adjust_lr = lambda step, epoch: lr_policy(step, epoch, initial_lrs,
        optimizer, steps_per_epoch=steps_per_epoch, warmup_epochs=args.
        warmup_epochs, hold_epochs=args.hold_epochs, min_lr=args.min_lr,
        exp_gamma=args.lr_exp_gamma, dist_lamb=args.dist_lamb)
    if not args.dist_lamb and multi_gpu:
        print('Using DDP here!')
        model = DistributedDataParallel(model)
    print_once('Setting up datasets...')
    (train_dataset_kw, train_features_kw, train_splicing_kw,
        train_padalign_kw, train_specaugm_kw) = config.input(cfg, 'train')
    (val_dataset_kw, val_features_kw, val_splicing_kw, val_padalign_kw,
        val_specaugm_kw) = config.input(cfg, 'val')
    logging.log_event(logging.constants.DATA_TRAIN_MAX_DURATION, value=
        train_dataset_kw['max_duration'])
    logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MAX, value=
        train_dataset_kw['speed_perturbation']['max_rate'])
    logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MIN, value=
        train_dataset_kw['speed_perturbation']['min_rate'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_N, value=
        train_specaugm_kw['freq_masks'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MIN, value=
        train_specaugm_kw['min_freq'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MAX, value=
        train_specaugm_kw['max_freq'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_N, value=
        train_specaugm_kw['time_masks'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MIN, value=
        train_specaugm_kw['min_time'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MAX, value=
        train_specaugm_kw['max_time'])
    logging.log_event(logging.constants.GLOBAL_BATCH_SIZE, value=batch_size *
        world_size * args.grad_accumulation_steps)


    class PermuteAudio(torch.nn.Module):

        def forward(self, x):
            return x[0].permute(2, 0, 1), *x[1:]
    if not train_specaugm_kw:
        train_specaugm = torch.nn.Identity()
    elif not args.vectorized_sa:
        train_specaugm = features.SpecAugment(optim_level=args.amp_level,
            **train_specaugm_kw)
    else:
        train_specaugm = features.VectorizedSpecAugment(optim_level=args.
            amp_level, **train_specaugm_kw)
    train_augmentations = torch.nn.Sequential(train_specaugm, features.
        FrameSplicing(optim_level=args.amp_level, **train_splicing_kw),
        features.FillPadding(optim_level=args.amp_level), features.PadAlign
        (optim_level=args.amp_level, **train_padalign_kw), PermuteAudio())
    if not val_specaugm_kw:
        val_specaugm = torch.nn.Identity()
    elif not args.vectorized_sa:
        val_specaugm = features.SpecAugment(optim_level=args.amp_level, **
            val_specaugm_kw)
    else:
        val_specaugm = features.VectorizedSpecAugment(optim_level=args.
            amp_level, **val_specaugm_kw)
    val_augmentations = torch.nn.Sequential(val_specaugm, features.
        FrameSplicing(optim_level=args.amp_level, **val_splicing_kw),
        features.FillPadding(optim_level=args.amp_level), features.PadAlign
        (optim_level=args.amp_level, **val_padalign_kw), PermuteAudio())
    train_feat_proc = train_augmentations
    val_feat_proc = val_augmentations
    train_feat_proc.cuda()
    val_feat_proc.cuda()
    train_preproc = Preproc(train_feat_proc, args.dist_lamb, args.
        apex_transducer_joint, args.batch_split_factor, cfg)
    if args.num_cg > 0:
        if not args.dist_lamb:
            raise NotImplementedError(
                'Currently CUDA graph training only works with dist LAMB')
        if args.batch_split_factor != 1:
            raise NotImplementedError(
                'Currently CUDA graph training does not work with batch split')
        max_seq_len = math.ceil(train_preproc.audio_duration_to_seq_len(cfg
            ['input_train']['audio_dataset']['max_duration'],
            after_subsampling=True, after_stack_time=False) * cfg[
            'input_train']['audio_dataset']['speed_perturbation']['max_rate'])
        print_once(f'Graph with max_seq_len of %d' % max_seq_len)
        rnnt_graph = RNNTGraph(model, rnnt_config, batch_size, max_seq_len,
            args.max_txt_len, args.num_cg)
        rnnt_graph.capture_graph()
    else:
        rnnt_graph = None
    if type(args.batch_eval_mode) == str and args.batch_eval_mode.startswith(
        'cg'):
        max_seq_len = train_preproc.audio_duration_to_seq_len(args.
            max_eval_sample_duration, after_subsampling=True,
            after_stack_time=True)
        dict_meta_data = {'batch': args.val_batch_size, 'max_feat_len':
            max_seq_len}
        greedy_decoder.capture_cg_for_eval(ema_model, dict_meta_data)

    def optimizer_warm_up():
        WARMUP_LEN = 8
        feats = torch.ones(WARMUP_LEN, batch_size, rnnt_config['in_feats'],
            dtype=torch.float16, device='cuda')
        feat_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda'
            ) * WARMUP_LEN
        txt = torch.ones(batch_size, WARMUP_LEN, dtype=torch.int64, device=
            'cuda')
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
    if args.dist_lamb:
        optimizer_warm_up()
    logging.log_end(logging.constants.INIT_STOP)
    if multi_gpu:
        dist.barrier()
    logging.log_start(logging.constants.RUN_START)
    if multi_gpu:
        dist.barrier()
    if args.pre_sort_for_seq_split and not args.vectorized_sampler:
        raise NotImplementedError(
            'Pre sort only works with vectorized sampler for now')
    logging.log_event(logging.constants.DATA_TRAIN_NUM_BUCKETS, value=args.
        num_buckets)
    if args.num_buckets is not None:
        if args.vectorized_sampler:
            builder = dali_sampler.VectorizedBucketingSampler
        else:
            builder = dali_sampler.BucketingSampler
        train_sampler = builder(train_dataset_kw, args.num_buckets,
            batch_size, world_size, args.epochs, sampler_seed, args.
            dist_sampler, args.pre_sort_for_seq_split)
    else:
        train_sampler = dali_sampler.SimpleSampler(train_dataset_kw)
    eval_sampler = dali_sampler.SimpleSampler(val_dataset_kw)
    train_sampler.sample(file_names=args.train_manifests, in_mem_file_list=
        args.in_mem_file_list, tokenized_transcript=args.tokenized_transcript)
    eval_sampler.sample(file_names=args.val_manifests, in_mem_file_list=
        args.in_mem_file_list, tokenized_transcript=args.tokenized_transcript)
    if (args.synthetic_audio_seq_len is None and args.
        synthetic_text_seq_len is None):
        synthetic_seq_len = None
    elif args.synthetic_audio_seq_len is not None and args.synthetic_text_seq_len is not None:
        synthetic_seq_len = [args.synthetic_audio_seq_len, args.
            synthetic_text_seq_len]
    else:
        raise Exception(
            'synthetic seq length for both text and audio need to be specified'
            )
    train_loader = DaliDataLoader(gpu_id=args.local_rank, dataset_path=args
        .dataset_dir, config_data=train_dataset_kw, config_features=
        train_features_kw, json_names=args.train_manifests, batch_size=
        batch_size, sampler=train_sampler, grad_accumulation_steps=args.
        grad_accumulation_steps, pipeline_type='train', device_type=args.
        dali_device, tokenizer=tokenizer, num_threads=args.data_cpu_threads,
        synthetic_seq_len=synthetic_seq_len, seed=dali_seed,
        in_mem_file_list=args.in_mem_file_list, enable_prefetch=args.
        enable_prefetch, tokenized_transcript=args.tokenized_transcript,
        preproc=train_preproc, min_seq_split_len=args.min_seq_split_len,
        pre_sort=args.pre_sort_for_seq_split, jit_tensor_formation=args.
        jit_tensor_formation, dont_use_mmap=args.dali_dont_use_mmap)
    val_loader = DaliDataLoader(gpu_id=args.local_rank, dataset_path=args.
        dataset_dir, config_data=val_dataset_kw, config_features=
        val_features_kw, json_names=args.val_manifests, batch_size=args.
        val_batch_size, sampler=eval_sampler, pipeline_type='val',
        device_type=args.dali_device, tokenizer=tokenizer, num_threads=args
        .data_cpu_threads, seed=dali_seed, tokenized_transcript=args.
        tokenized_transcript, in_mem_file_list=args.in_mem_file_list,
        dont_use_mmap=args.dali_dont_use_mmap)
    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps
    logging.log_event(logging.constants.TRAIN_SAMPLES, value=train_loader.
        dataset_size)
    logging.log_event(logging.constants.EVAL_SAMPLES, value=val_loader.
        dataset_size)
    logging.log_event(logging.constants.OPT_NAME, value='lamb')
    logging.log_event(logging.constants.OPT_BASE_LR, value=args.lr)
    logging.log_event(logging.constants.OPT_LAMB_EPSILON, value=opt_eps)
    logging.log_event(logging.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value
        =args.lr_exp_gamma)
    logging.log_event(logging.constants.OPT_LR_WARMUP_EPOCHS, value=args.
        warmup_epochs)
    logging.log_event(logging.constants.OPT_LAMB_LR_HOLD_EPOCHS, value=args
        .hold_epochs)
    logging.log_event(logging.constants.OPT_LAMB_BETA_1, value=args.beta1)
    logging.log_event(logging.constants.OPT_LAMB_BETA_2, value=args.beta2)
    logging.log_event(logging.constants.OPT_GRADIENT_CLIP_NORM, value=args.
        clip_norm)
    logging.log_event(logging.constants.OPT_LR_ALT_DECAY_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LR_ALT_WARMUP_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LAMB_LR_MIN, value=args.min_lr)
    logging.log_event(logging.constants.OPT_WEIGHT_DECAY, value=args.
        weight_decay)
    meta = {'best_wer': 10 ** 6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'RNN-T', args.
        keep_milestones, use_amp=True)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt
    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)
    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    last_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1
    model.train()
    if args.multi_tensor_ema == True:
        ema_model_weight_list, model_weight_list, overflow_buf_for_ema = (
            init_multi_tensor_ema(optimizer, model, ema_model, args.
            ema_update_type))
    copy_stream = torch.cuda.Stream()
    pred_stream = torch.cuda.Stream()

    def buffer_pre_allocation():
        max_seq_len = math.ceil(train_preproc.audio_duration_to_seq_len(cfg
            ['input_train']['audio_dataset']['max_duration'],
            after_subsampling=False, after_stack_time=False) * cfg[
            'input_train']['audio_dataset']['speed_perturbation']['max_rate'])
        max_txt_len = train_loader.data_iterator().max_txt_len
        print_once(
            f'Pre-allocate buffer with max_seq_len of %d and max_txt_len of %d'
             % (max_seq_len, max_txt_len))
        audio = torch.ones(batch_size, cfg['input_val'][
            'filterbank_features']['n_filt'], max_seq_len, dtype=torch.
            float32, device='cuda')
        audio_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda'
            ) * max_seq_len
        txt = torch.ones(batch_size, max_txt_len, dtype=torch.int64, device
            ='cuda')
        txt_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda'
            ) * max_txt_len
        feats, feat_lens = train_feat_proc([audio, audio_lens])
        if args.dist_lamb:
            feats = feats.half()
        meta_data = []
        B_split = batch_size // args.batch_split_factor
        for i in range(args.batch_split_factor):
            meta_data.append(train_preproc.get_packing_meta_data(feats.size
                (0), feat_lens[i * B_split:(i + 1) * B_split], txt_lens[i *
                B_split:(i + 1) * B_split]))
        train_step(model, loss_fn, args, batch_size, feats, feat_lens, txt,
            txt_lens, optimizer, grad_scaler, meta_data, None, rnnt_graph,
            copy_stream, pred_stream)
    if args.buffer_pre_alloc:
        buffer_pre_allocation()
    training_start_time = time.time()
    training_utts = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        logging.log_start(logging.constants.BLOCK_START, metadata=dict(
            first_epoch_num=epoch, epoch_count=1))
        logging.log_start(logging.constants.EPOCH_START, metadata=dict(
            epoch_num=epoch))
        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()
        if args.enable_prefetch:
            train_loader.data_iterator().prefetch()
        step_start_time = time.time()
        for batch in train_loader:
            if accumulated_batches == 0:
                if not args.dist_lamb:
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
                adjust_lr(step, epoch)
                step_utts = 0
                all_feat_lens = []
            if args.enable_prefetch:
                feats, feat_lens, txt, txt_lens = batch
                meta_data = train_preproc.meta_data
            else:
                audio, audio_lens, txt, txt_lens = batch
                feats, feat_lens = train_feat_proc([audio, audio_lens])
                if args.dist_lamb:
                    feats = feats.half()
                meta_data = []
                B_split = batch_size // args.batch_split_factor
                for i in range(args.batch_split_factor):
                    meta_data.append(train_preproc.get_packing_meta_data(
                        feats.size(0), feat_lens[i * B_split:(i + 1) *
                        B_split], txt_lens[i * B_split:(i + 1) * B_split]))
            if args.enable_seq_len_stats:
                all_feat_lens += feat_lens
            loss_item, lr_item = train_step(model, loss_fn, args,
                batch_size, feats, feat_lens, txt, txt_lens, optimizer,
                grad_scaler, meta_data, train_loader, rnnt_graph,
                copy_stream, pred_stream)
            step_utts += txt_lens.size(0) * world_size
            epoch_utts += txt_lens.size(0) * world_size
            accumulated_batches += 1
            if accumulated_batches % args.grad_accumulation_steps == 0:
                if args.dist_lamb:
                    optimizer.complete_reductions()
                    optimizer.set_global_scale(grad_scaler._get_scale_async())
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                if args.multi_tensor_ema == True:
                    apply_multi_tensor_ema(model_weight_list,
                        ema_model_weight_list, args.ema, overflow_buf_for_ema)
                else:
                    apply_ema(optimizer, ema_model, args.ema)
                if step % args.log_frequency == 0:
                    if (args.prediction_frequency is None or step % args.
                        prediction_frequency == 0):
                        preds = greedy_decoder.decode(feats.half(), feat_lens)
                        wer, pred_utt, ref = greedy_wer(preds, txt,
                            txt_lens, tokenizer.detokenize)
                        print_once(f'  Decoded:   {pred_utt[:90]}')
                        print_once(f'  Reference: {ref[:90]}')
                        wer = {'wer': 100 * wer}
                    else:
                        wer = {}
                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    dict_log = {'loss': loss_item, **wer, 'throughput': 
                        step_utts / step_time, 'took': step_time, 'lrate': 
                        optimizer._lr.item() if args.dist_lamb else
                        optimizer.param_groups[0]['lr']}
                    if args.enable_seq_len_stats:
                        dict_log['seq-len-min'] = min(all_feat_lens).item()
                        dict_log['seq-len-max'] = max(all_feat_lens).item()
                    log((epoch, step % steps_per_epoch or steps_per_epoch,
                        steps_per_epoch), step, 'train', dict_log)
                step += 1
                accumulated_batches = 0
        logging.log_end(logging.constants.EPOCH_STOP, metadata=dict(
            epoch_num=epoch))
        epoch_time = time.time() - epoch_start_time
        log((epoch,), None, 'train_avg', {'throughput': epoch_utts /
            epoch_time, 'took': epoch_time})
        logging.log_event(key='throughput', value=epoch_utts / epoch_time)
        if epoch % args.val_frequency == 0:
            wer = evaluate(epoch, step, val_loader, val_feat_proc,
                tokenizer.detokenize, ema_model, loss_fn, greedy_decoder,
                args.amp_level)
            last_wer = wer
            if wer < best_wer and epoch >= args.save_best_from:
                checkpointer.save(model, ema_model, optimizer, epoch, step,
                    best_wer, is_best=True)
                best_wer = wer
        save_this_epoch = (args.save_frequency is not None and epoch % args
            .save_frequency == 0 or epoch in args.keep_milestones)
        if save_this_epoch:
            checkpointer.save(model, ema_model, optimizer, epoch, step,
                best_wer)
        training_utts += epoch_utts
        logging.log_end(logging.constants.BLOCK_STOP, metadata=dict(
            first_epoch_num=epoch))
        if last_wer <= args.target:
            logging.log_end(logging.constants.RUN_STOP, metadata={'status':
                'success'})
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
    training_time = time.time() - training_start_time
    log((), None, 'train_avg', {'throughput': training_utts / training_time})
    if last_wer > args.target:
        logging.log_end(logging.constants.RUN_STOP, metadata={'status':
            'aborted'})
    if epoch == args.epochs:
        print('Evaluating the model at the end')
        evaluate(epoch, step, val_loader, val_feat_proc, tokenizer.
            detokenize, ema_model, loss_fn, greedy_decoder, args.amp_level)
    flush_log()
    if args.save_at_the_end:
        print('Saving the model at the end')
        checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)
