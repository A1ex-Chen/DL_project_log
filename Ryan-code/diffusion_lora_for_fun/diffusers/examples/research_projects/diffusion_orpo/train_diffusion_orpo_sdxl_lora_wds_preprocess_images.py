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
        y1 = max(0, int(round((combined_im.shape[1] - args.resolution) / 2.0)))
        x1 = max(0, int(round((combined_im.shape[2] - args.resolution) / 2.0)))
        combined_im = train_crop(combined_im)
    else:
        y1, x1, h, w = train_crop.get_params(combined_im, (args.resolution,
            args.resolution))
        combined_im = crop(combined_im, y1, x1, h, w)
    crop_top_left = y1, x1
    combined_im = normalize(combined_im)
    tokens_one, tokens_two = tokenize_captions([tokenizer_one,
        tokenizer_two], sample)
    return {'pixel_values': combined_im, 'original_size': original_size,
        'crop_top_left': crop_top_left, 'tokens_one': tokens_one,
        'tokens_two': tokens_two}
