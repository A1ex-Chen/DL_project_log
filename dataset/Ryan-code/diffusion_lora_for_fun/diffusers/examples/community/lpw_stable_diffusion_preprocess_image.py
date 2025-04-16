def preprocess_image(image, batch_size):
    w, h = image.size
    w, h = (x - x % 8 for x in (w, h))
    image = image.resize((w, h), resample=PIL_INTERPOLATION['lanczos'])
    image = np.array(image).astype(np.float32) / 255.0
    image = np.vstack([image[None].transpose(0, 3, 1, 2)] * batch_size)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0
