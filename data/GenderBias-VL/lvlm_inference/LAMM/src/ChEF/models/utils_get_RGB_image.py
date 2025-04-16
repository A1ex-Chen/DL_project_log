def get_RGB_image(image):
    image = get_image(image)
    image = Image.fromarray(np.uint8(image))
    return image
