def get_train_dataset(args, accelerator):
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name,
            cache_dir=args.cache_dir)
    elif args.train_data_dir is not None:
        dataset = load_dataset(args.train_data_dir, cache_dir=args.cache_dir)
    column_names = dataset['train'].column_names
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f'image column defaulting to {image_column}')
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f'caption column defaulting to {caption_column}')
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(
            f'conditioning image column defaulting to {conditioning_image_column}'
            )
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
    with accelerator.main_process_first():
        train_dataset = dataset['train'].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset
