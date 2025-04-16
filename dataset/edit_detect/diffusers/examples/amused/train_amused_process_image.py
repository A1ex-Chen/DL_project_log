def process_image(image, size):
    image = exif_transpose(image)
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    orig_height = image.height
    orig_width = image.width
    image = transforms.Resize(size, interpolation=transforms.
        InterpolationMode.BILINEAR)(image)
    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image,
        output_size=(size, size))
    image = transforms.functional.crop(image, c_top, c_left, size, size)
    image = transforms.ToTensor()(image)
    micro_conds = torch.tensor([orig_width, orig_height, c_top, c_left, 6.0])
    return {'image': image, 'micro_conds': micro_conds}
