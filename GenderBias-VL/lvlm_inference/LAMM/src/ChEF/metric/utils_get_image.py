def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert('RGB')
        except Exception as e:
            print(f'Fail to read image: {image}')
            exit(-1)
    elif isinstance(image, Image.Image):
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        raise NotImplementedError(f'Invalid type of Image: {type(image)}')
