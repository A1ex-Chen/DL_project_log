def preprocess_train(examples):
    images = [image.convert('RGB') for image in examples[image_column]]
    examples['effnet_pixel_values'] = [effnet_transforms(image) for image in
        images]
    examples['text_input_ids'], examples['text_mask'] = tokenize_captions(
        examples)
    return examples
