def preprocess_train(examples):
    images = [image.convert('RGB') for image in examples[image_column]]
    examples['pixel_values'] = [train_transforms(image) for image in images]
    examples['clip_pixel_values'] = image_processor(images, return_tensors='pt'
        ).pixel_values
    return examples
