def normalize_images(images: List[Image.Image]):
    images = [np.array(image) for image in images]
    images = [(image / 127.5 - 1) for image in images]
    return images
