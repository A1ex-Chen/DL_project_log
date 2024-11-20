def process_images(images, image_processor, model_cfg=None):
    if model_cfg is not None:
        image_aspect_ratio = getattr(model_cfg, 'image_aspect_ratio', None)
    else:
        image_aspect_ratio = 'resize'
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in
                image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == 'resize':
        for image in images:
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            image = image_processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images
