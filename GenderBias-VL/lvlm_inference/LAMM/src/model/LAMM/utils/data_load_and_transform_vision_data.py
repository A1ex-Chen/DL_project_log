def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None
    image_ouputs = []
    for image_path in image_paths:
        data_transform = transforms.Compose([transforms.Resize(224,
            interpolation=transforms.InterpolationMode.BICUBIC), transforms
            .CenterCrop(224), transforms.ToTensor(), transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 
            0.26130258, 0.27577711))])
        if os.path.exists(image_path):
            with open(image_path, 'rb') as fopen:
                image = Image.open(fopen).convert('RGB')
        elif image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw
                ).convert('RGB')
        else:
            raise ValueError(f'Invalid image path: {image_path}')
        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)
