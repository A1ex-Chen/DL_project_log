def convert_to_np(image, resolution):
    image = image.convert('RGB').resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)
