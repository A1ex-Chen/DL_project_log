def __init__(self, image_size=364, mean=None, std=None, min_scale=0.5,
    max_scale=1.0):
    super().__init__(mean=mean, std=std)
    self.transform = transforms.Compose([transforms.RandomResizedCrop(
        image_size, scale=(min_scale, max_scale), interpolation=
        InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), self.normalize])
