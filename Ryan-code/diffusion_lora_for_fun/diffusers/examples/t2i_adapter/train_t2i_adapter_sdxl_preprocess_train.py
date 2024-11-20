def preprocess_train(examples):
    images = [image.convert('RGB') for image in examples[args.image_column]]
    images = [image_transforms(image) for image in images]
    conditioning_images = [image.convert('RGB') for image in examples[args.
        conditioning_image_column]]
    conditioning_images = [conditioning_image_transforms(image) for image in
        conditioning_images]
    examples['pixel_values'] = images
    examples['conditioning_pixel_values'] = conditioning_images
    return examples
