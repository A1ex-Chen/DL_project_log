def preprocess(image):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return 2.0 * image - 1.0
