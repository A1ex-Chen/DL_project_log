def preprocess(self, img):
    """Converts input image to model-compatible data type."""
    if not isinstance(img, torch.Tensor):
        is_legacy_transform = any(self._legacy_transform_name in str(
            transform) for transform in self.transforms.transforms)
        if is_legacy_transform:
            img = torch.stack([self.transforms(im) for im in img], dim=0)
        else:
            img = torch.stack([self.transforms(Image.fromarray(cv2.cvtColor
                (im, cv2.COLOR_BGR2RGB))) for im in img], dim=0)
    img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(
        self.model.device)
    return img.half() if self.model.fp16 else img.float()
