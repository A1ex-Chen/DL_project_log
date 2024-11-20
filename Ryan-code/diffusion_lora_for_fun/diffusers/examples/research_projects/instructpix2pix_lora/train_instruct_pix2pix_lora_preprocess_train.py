def preprocess_train(examples):
    preprocessed_images = preprocess_images(examples)
    original_images, edited_images = preprocessed_images.chunk(2)
    original_images = original_images.reshape(-1, 3, args.resolution, args.
        resolution)
    edited_images = edited_images.reshape(-1, 3, args.resolution, args.
        resolution)
    examples['original_pixel_values'] = original_images
    examples['edited_pixel_values'] = edited_images
    captions = list(examples[edit_prompt_column])
    examples['input_ids'] = tokenize_captions(captions)
    return examples
