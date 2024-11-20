def preprocess_train(examples):
    images = [image.convert('RGB') for image in examples[image_column]]
    examples['clip_pixel_values'] = image_processor(images, return_tensors='pt'
        ).pixel_values
    examples['text_input_ids'], examples['text_mask'] = tokenize_captions(
        examples)
    return examples
