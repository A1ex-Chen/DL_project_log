def encode(self, image_paths: List[str]):
    images = []
    for image_path in image_paths:
        if image_path.startswith('http://') or image_path.startswith('https://'
            ):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        image = image.convert('RGB')
        images.append(self.image_transform(image))
    images = torch.stack(images, dim=0)
    return self(images)
