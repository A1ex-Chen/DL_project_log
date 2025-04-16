def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search('yolov\\d+([nslmx])', Path(model_path).stem).group(1)
    return ''
