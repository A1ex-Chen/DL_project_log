def transform_vision_data(self, images, device):
    image_ouputs = []
    for img in images:
        data_transform = transforms.Compose([transforms.Resize(224,
            interpolation=transforms.InterpolationMode.BICUBIC), transforms
            .CenterCrop(224), transforms.ToTensor(), transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 
            0.26130258, 0.27577711))])
        image = data_transform(img).to(device)
        image = image.half()
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)
