def process_image(img_src, img_size, stride):
    """Process image before image inference."""
    image = letterbox(img_src, img_size, auto=False)[0]
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image).astype(np.float32)
    image /= 255.0
    return image
