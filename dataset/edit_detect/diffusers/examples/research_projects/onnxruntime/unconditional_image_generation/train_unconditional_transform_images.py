def transform_images(examples):
    images = [augmentations(image.convert('RGB')) for image in examples[
        'image']]
    return {'input': images}
