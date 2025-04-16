def preprocess_image(image):
    """
    image: torch.Tensor
    """
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))
    image = image.resize((w, h))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).contiguous()
    return 2.0 * image - 1.0
