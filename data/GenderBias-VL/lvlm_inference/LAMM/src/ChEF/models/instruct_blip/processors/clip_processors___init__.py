def __init__(self, image_size=224, mean=None, std=None):
    super().__init__(mean=mean, std=std)
    self.transform = transforms.Compose([transforms.Resize(image_size,
        interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(
        image_size), _convert_to_rgb, transforms.ToTensor(), self.normalize])
