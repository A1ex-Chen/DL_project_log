def get_loader(args, tokenizer_one, tokenizer_two):
    num_batches = math.ceil(args.num_train_examples / args.global_batch_size)
    num_worker_batches = math.ceil(args.num_train_examples / (args.
        global_batch_size * args.dataloader_num_workers))
    num_batches = num_worker_batches * args.dataloader_num_workers
    num_samples = num_batches * args.global_batch_size
    dataset = get_dataset(args)
    train_resize = transforms.Resize(args.resolution, interpolation=
        transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.RandomCrop(args.resolution
        ) if args.random_crop else transforms.CenterCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])

    def preprocess_images(sample):
        jpg_0_image = sample['jpg_0']
        original_size = jpg_0_image.height, jpg_0_image.width
        crop_top_left = []
        jpg_1_image = sample['jpg_1']
        jpg_1_image = jpg_1_image.resize(original_size[::-1])
        label_0 = sample['label_0']
        if sample['label_0'] == 0.5:
            if random.random() < 0.5:
                label_0 = 0
            else:
                label_0 = 1
        if label_0 == 0:
            pixel_values = torch.cat([to_tensor(image) for image in [
                jpg_1_image, jpg_0_image]])
        else:
            pixel_values = torch.cat([to_tensor(image) for image in [
                jpg_0_image, jpg_1_image]])
        combined_im = train_resize(pixel_values)
        if not args.no_hflip and random.random() < 0.5:
            combined_im = train_flip(combined_im)
        if not args.random_crop:
            y1 = max(0, int(round((combined_im.shape[1] - args.resolution) /
                2.0)))
            x1 = max(0, int(round((combined_im.shape[2] - args.resolution) /
                2.0)))
            combined_im = train_crop(combined_im)
        else:
            y1, x1, h, w = train_crop.get_params(combined_im, (args.
                resolution, args.resolution))
            combined_im = crop(combined_im, y1, x1, h, w)
        crop_top_left = y1, x1
        combined_im = normalize(combined_im)
        tokens_one, tokens_two = tokenize_captions([tokenizer_one,
            tokenizer_two], sample)
        return {'pixel_values': combined_im, 'original_size': original_size,
            'crop_top_left': crop_top_left, 'tokens_one': tokens_one,
            'tokens_two': tokens_two}

    def collate_fn(samples):
        pixel_values = torch.stack([sample['pixel_values'] for sample in
            samples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format
            ).float()
        original_sizes = [example['original_size'] for example in samples]
        crop_top_lefts = [example['crop_top_left'] for example in samples]
        input_ids_one = torch.stack([example['tokens_one'] for example in
            samples])
        input_ids_two = torch.stack([example['tokens_two'] for example in
            samples])
        return {'pixel_values': pixel_values, 'input_ids_one':
            input_ids_one, 'input_ids_two': input_ids_two, 'original_sizes':
            original_sizes, 'crop_top_lefts': crop_top_lefts}
    dataset = dataset.map(preprocess_images, handler=wds.warn_and_continue)
    dataset = dataset.batched(args.per_gpu_batch_size, partial=False,
        collation_fn=collate_fn)
    dataset = dataset.with_epoch(num_worker_batches)
    dataloader = wds.WebLoader(dataset, batch_size=None, shuffle=False,
        num_workers=args.dataloader_num_workers, pin_memory=True,
        persistent_workers=True)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return dataloader
