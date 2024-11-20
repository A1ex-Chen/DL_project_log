def generate_image():
    return PIL.Image.fromarray(np.random.randint(0, 256, size=(height,
        width, num_channels)).astype('uint8'))
