def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, 'image_aspect_ratio', None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in
                image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images
