def preprocess_train(examples):
    images = [image.convert('RGB') for image in examples[image_column]]
    examples['pixel_values'] = [train_transforms(image) for image in images]
    examples['input_ids'] = tokenize_captions(examples)
    return examples
