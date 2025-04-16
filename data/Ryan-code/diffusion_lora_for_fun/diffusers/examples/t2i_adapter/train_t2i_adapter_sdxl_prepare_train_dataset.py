def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose([transforms.Resize(args.
        resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution), transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    conditioning_image_transforms = transforms.Compose([transforms.Resize(
        args.resolution, interpolation=transforms.InterpolationMode.
        BILINEAR), transforms.CenterCrop(args.resolution), transforms.
        ToTensor()])

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[args.image_column]
            ]
        images = [image_transforms(image) for image in images]
        conditioning_images = [image.convert('RGB') for image in examples[
            args.conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for
            image in conditioning_images]
        examples['pixel_values'] = images
        examples['conditioning_pixel_values'] = conditioning_images
        return examples
    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)
    return dataset
