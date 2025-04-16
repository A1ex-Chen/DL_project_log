def get_csv_dataset(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(input_filename, preprocess_fn, img_key=args.
        csv_img_key, caption_key=args.csv_caption_key, sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset
        ) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=
        shuffle, num_workers=args.workers, pin_memory=True, sampler=sampler,
        drop_last=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)
