def get_wds_dataset(args, model_cfg, is_train, audio_ext='flac', text_ext=
    'json', max_len=480000, proportion=1.0, sizefilepath_=None, is_local=None):
    """
    Get a dataset for wdsdataloader.
    """
    if is_local is None and not args.remotedata is None:
        is_local = not args.remotedata
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    if not sizefilepath_ is None:
        sizefilepath = sizefilepath_
    else:
        sizefilepath = os.path.join(os.path.dirname(input_shards[0]),
            'sizes.json')
    if proportion != 1.0:
        num_samples, num_shards, input_shards, _ = sample_prop(sizefilepath,
            input_shards, proportion, is_local=is_local)
    else:
        num_samples, num_shards = get_dataset_size(input_shards,
            sizefilepath_=sizefilepath_, is_local=is_local)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. Please specify via `--train-num-samples` if no dataset length info present.'
                    )
        else:
            num_samples = args.val_num_samples or 0
    pipeline = [wds.SimpleShardList(input_shards)]
    if is_train or args.parallel_eval:
        pipeline.extend([wds.detshuffle(bufsize=_SHARD_SHUFFLE_SIZE,
            initial=_SHARD_SHUFFLE_INITIAL, seed=args.seed), wds.
            split_by_node, wds.split_by_worker, wds.tarfile_to_samples(
            handler=log_and_continue), wds.shuffle(bufsize=
            _SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL, rng=
            random.Random(args.seed))])
    else:
        pipeline.extend([wds.split_by_worker, wds.tarfile_to_samples(
            handler=log_and_continue)])
    pipeline.append(wds.decode(wds.torch_audio))
    pipeline.append(wds.batched(args.batch_size, partial=not (is_train or
        args.parallel_eval), collation_fn=partial(
        collate_fn_with_preprocess, audio_ext=audio_ext, text_ext=text_ext,
        max_len=max_len, audio_cfg=model_cfg['audio_cfg'], args=args)))
    dataset = wds.DataPipeline(*pipeline)
    if is_train or args.parallel_eval:
        global_batch_size = args.batch_size * args.world_size
        num_batches = math.ceil(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = math.ceil(num_batches / num_workers)
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)
    else:
        num_batches = math.ceil(num_samples / args.batch_size)
    kwargs = {}
    if args.horovod:
        kwargs['multiprocessing_context'] = 'forkserver'
    if is_train:
        if args.prefetch_factor:
            prefetch_factor = args.prefetch_factor
        else:
            prefetch_factor = max(2, args.batch_size // args.workers)
    else:
        prefetch_factor = 2
    dataloader = wds.WebLoader(dataset, batch_size=None, shuffle=False,
        num_workers=args.workers, pin_memory=True, prefetch_factor=
        prefetch_factor, **kwargs)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader, None)
