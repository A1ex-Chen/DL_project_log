def to_np(self, image):
    if isinstance(image[0], PIL.Image.Image):
        return np.stack([np.array(i) for i in image], axis=0)
    elif isinstance(image, torch.Tensor):
        return image.cpu().numpy().transpose(0, 2, 3, 1)
    return image
