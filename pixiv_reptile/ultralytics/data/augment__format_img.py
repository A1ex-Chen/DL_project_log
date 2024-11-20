def _format_img(self, img):
    """Format the image for YOLO from Numpy array to PyTorch tensor."""
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr
         else img)
    img = torch.from_numpy(img)
    return img
