def __call__(self, img):
    """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
    h = img.size(1)
    w = img.size(2)
    mask = np.ones((h, w), np.float32)
    for _ in range(self.n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask
    return img
