def apply_watermark(self, images: torch.Tensor):
    if images.shape[-1] < 256:
        return images
    images = (255 * (images / 2 + 0.5)).cpu().permute(0, 2, 3, 1).float(
        ).numpy()
    images = images[:, :, :, ::-1]
    images = [self.encoder.encode(image, 'dwtDct')[:, :, ::-1] for image in
        images]
    images = np.array(images)
    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    images = torch.clamp(2 * (images / 255 - 0.5), min=-1.0, max=1.0)
    return images
