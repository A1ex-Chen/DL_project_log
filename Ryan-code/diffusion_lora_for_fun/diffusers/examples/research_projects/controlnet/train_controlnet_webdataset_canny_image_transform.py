def canny_image_transform(example, resolution=1024):
    image = example['image']
    image = transforms.Resize(resolution, interpolation=transforms.
        InterpolationMode.BILINEAR)(image)
    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image,
        output_size=(resolution, resolution))
    image = transforms.functional.crop(image, c_top, c_left, resolution,
        resolution)
    control_image = control_transform(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.5], [0.5])(image)
    control_image = transforms.ToTensor()(control_image)
    example['image'] = image
    example['control_image'] = control_image
    example['crop_coords'] = c_top, c_left
    return example
