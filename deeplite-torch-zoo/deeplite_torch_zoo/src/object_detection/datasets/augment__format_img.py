def _format_img(self, img):
    """Format the image for YOLOv5 from Numpy array to PyTorch tensor."""
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
    img = torch.from_numpy(img)
    return img
