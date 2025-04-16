def get_toy_dataset(args, model_cfg, is_train):
    index_path = args.train_data if is_train else args.val_data
    ipc_path = args.train_ipc if is_train else args.val_ipc
    assert index_path and ipc_path
    eval_mode = not is_train
    dataset = ToyDataset(index_path, ipc_path, model_cfg, eval_mode=eval_mode)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset, shuffle=False
        ) if args.distributed and is_train else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=
        False, num_workers=args.workers, sampler=sampler, drop_last=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)
