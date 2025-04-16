def make_train_dataset(args, tokenizer, accelerator):
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

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append('')
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else
                    caption[0])
            else:
                raise ValueError(
                    f'Caption column `{caption_column}` should contain either strings or lists of strings.'
                    )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length,
            padding='max_length', truncation=True, return_tensors='pt')
        return inputs.input_ids
    image_transforms = transforms.Compose([transforms.Resize(args.
        resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution), transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    conditioning_image_transforms = transforms.Compose([transforms.Resize(
        args.resolution, interpolation=transforms.InterpolationMode.
        BILINEAR), transforms.CenterCrop(args.resolution), transforms.
        ToTensor()])

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[image_column]]
        images = [image_transforms(image) for image in images]
        conditioning_images = [image.convert('RGB') for image in examples[
            conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for
            image in conditioning_images]
        examples['pixel_values'] = images
        examples['conditioning_pixel_values'] = conditioning_images
        examples['input_ids'] = tokenize_captions(examples)
        return examples
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(
                range(args.max_train_samples))
        train_dataset = dataset['train'].with_transform(preprocess_train)
    return train_dataset
