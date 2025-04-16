def preprocess_train(examples):
    images = [image.convert('RGB') for image in examples[image_column]]
    original_sizes = []
    all_images = []
    crop_top_lefts = []
    for image in images:
        original_sizes.append((image.height, image.width))
        image = train_resize(image)
        if args.center_crop:
            y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
            image = train_crop(image)
        else:
            y1, x1, h, w = train_crop.get_params(image, (args.resolution,
                args.resolution))
            image = crop(image, y1, x1, h, w)
        if args.random_flip and random.random() < 0.5:
            x1 = image.width - x1
            image = train_flip(image)
        crop_top_left = y1, x1
        crop_top_lefts.append(crop_top_left)
        image = train_transforms(image)
        all_images.append(image)
    examples['original_sizes'] = original_sizes
    examples['crop_top_lefts'] = crop_top_lefts
    examples['pixel_values'] = all_images
    examples['captions'] = list(examples[caption_column])
    return examples
