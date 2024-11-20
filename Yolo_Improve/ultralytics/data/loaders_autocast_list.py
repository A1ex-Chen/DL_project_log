def autocast_list(source):
    """Merges a list of source of different types into a list of numpy arrays or PIL images."""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):
            files.append(Image.open(requests.get(im, stream=True).raw if
                str(im).startswith('http') else im))
        elif isinstance(im, (Image.Image, np.ndarray)):
            files.append(im)
        else:
            raise TypeError(
                f"""type {type(im).__name__} is not a supported Ultralytics prediction source type. 
See https://docs.ultralytics.com/modes/predict for supported source types."""
                )
    return files
