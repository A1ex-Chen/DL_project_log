def process_image(path, img_size, stride):
    """Preprocess image before inference."""
    try:
        img_src = cv2.imread(path)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        assert img_src is not None, f'opencv cannot read image correctly or {path} not exists'
    except:
        img_src = np.asarray(Image.open(path))
        assert img_src is not None, f'Image Not Found {path}, workdir: {os.getcwd()}'
    image = letterbox(img_src, img_size, stride=stride)[0]
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.float()
    image /= 255
    return image, img_src
