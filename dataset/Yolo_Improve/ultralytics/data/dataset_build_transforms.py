def build_transforms(self, hyp=None):
    """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity."""
    transforms = super().build_transforms(hyp)
    if self.augment:
        transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))
    return transforms
