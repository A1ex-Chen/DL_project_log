def get_dataloaders(args, accelerator):
    data_files = {}
    if args.train_file is not None:
        data_files['train'] = args.train_file
    if args.validation_file is not None:
        data_files['validation'] = args.validation_file
    if args.test_file is not None:
        data_files['test'] = args.test_file
    elif args.validation_file is not None:
        data_files['test'] = args.validation_file
    extension = args.train_file.split('.')[-1]
    raw_datasets = datasets.load_dataset(extension, data_files=data_files)
    text_column, audio_column = args.text_column, args.audio_column
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ''
    with accelerator.main_process_first():
        train_dataset = Text2AudioDataset(raw_datasets['train'], prefix,
            text_column, audio_column, args.num_examples, target_length=
            TARGET_LENGTH, augment=True)
        eval_dataset = Text2AudioDataset(raw_datasets['validation'], prefix,
            text_column, audio_column, args.num_examples, target_length=
            TARGET_LENGTH, augment=False)
        test_dataset = Text2AudioDataset(raw_datasets['test'], prefix,
            text_column, audio_column, args.num_examples, target_length=
            TARGET_LENGTH, augment=False)
        logger.info(
            f'Num instances in train: {len(train_dataset)}, validation: {len(eval_dataset)}, test: {len(test_dataset)}.'
            )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=
        args.per_device_train_batch_size, num_workers=4, collate_fn=
        train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=
        args.per_device_eval_batch_size, num_workers=4, collate_fn=
        eval_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=
        args.per_device_eval_batch_size, num_workers=4, collate_fn=
        test_dataset.collate_fn)
    return train_dataloader, eval_dataloader, test_dataloader
