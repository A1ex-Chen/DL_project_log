def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = (False, False, False,
        False, False)
    if isinstance(source, (str, int, Path)):
        source = str(source)
        is_file = Path(source).suffix[1:] in IMG_FORMATS | VID_FORMATS
        is_url = source.lower().startswith(('https://', 'http://',
            'rtsp://', 'rtmp://', 'tcp://'))
        webcam = source.isnumeric() or source.endswith('.streams'
            ) or is_url and not is_file
        screenshot = source.lower() == 'screen'
        if is_url and is_file:
            source = check_file(source)
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError(
            'Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict'
            )
    return source, webcam, screenshot, from_img, in_memory, tensor
