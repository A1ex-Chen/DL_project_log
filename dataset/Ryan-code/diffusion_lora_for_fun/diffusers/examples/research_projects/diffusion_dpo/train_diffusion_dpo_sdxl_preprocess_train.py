def preprocess_train(examples):
    all_pixel_values = []
    images = [Image.open(io.BytesIO(im_bytes)).convert('RGB') for im_bytes in
        examples['jpg_0']]
    original_sizes = [(image.height, image.width) for image in images]
    crop_top_lefts = []
    for col_name in ['jpg_0', 'jpg_1']:
        images = [Image.open(io.BytesIO(im_bytes)).convert('RGB') for
            im_bytes in examples[col_name]]
        if col_name == 'jpg_1':
            images = [image.resize(original_sizes[i][::-1]) for i, image in
                enumerate(images)]
        pixel_values = [to_tensor(image) for image in images]
        all_pixel_values.append(pixel_values)
    im_tup_iterator = zip(*all_pixel_values)
    combined_pixel_values = []
    for im_tup, label_0 in zip(im_tup_iterator, examples['label_0']):
        if label_0 == 0:
            im_tup = im_tup[::-1]
        combined_im = torch.cat(im_tup, dim=0)
        combined_im = train_resize(combined_im)
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
        crop_top_lefts.append(crop_top_left)
        combined_im = normalize(combined_im)
        combined_pixel_values.append(combined_im)
    examples['pixel_values'] = combined_pixel_values
    examples['original_sizes'] = original_sizes
    examples['crop_top_lefts'] = crop_top_lefts
    tokens_one, tokens_two = tokenize_captions([tokenizer_one,
        tokenizer_two], examples)
    examples['input_ids_one'] = tokens_one
    examples['input_ids_two'] = tokens_two
    return examples
