def convert_to_np(image, resolution):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert('RGB').resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)
