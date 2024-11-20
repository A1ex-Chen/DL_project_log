def __init__(self, root, augment, imgsz, cache=False):
    super().__init__(root=root)
    self.torch_transforms = classify_transforms(imgsz)
    self.album_transforms = classify_albumentations(augment, imgsz
        ) if augment else None
    self.cache_ram = cache is True or cache == 'ram'
    self.cache_disk = cache == 'disk'
    self.samples = [(list(x) + [Path(x[0]).with_suffix('.npy'), None]) for
        x in self.samples]
