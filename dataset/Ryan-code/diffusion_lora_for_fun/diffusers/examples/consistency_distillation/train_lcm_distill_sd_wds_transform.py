def transform(example):
    image = example['image']
    image = TF.resize(image, resolution, interpolation=interpolation_mode)
    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image,
        output_size=(resolution, resolution))
    image = TF.crop(image, c_top, c_left, resolution, resolution)
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])
    example['image'] = image
    return example
