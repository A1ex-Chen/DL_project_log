@staticmethod
def process_image(img_src, img_size, stride, half):
    """Process image before image inference."""
    image = letterbox(img_src, img_size, stride=stride)[0]
    image = image.transpose((2, 0, 1))[::-1]
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()
    image /= 255
    return image, img_src
