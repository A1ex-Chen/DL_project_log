def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]
    image = [trans(img.convert('RGB')) for img in image]
    image = torch.stack(image)
    return image
