def infer_bbox_format(filename: PathLike) ->str:
    """Try to infer the format of a given bounding box file.

    Args:
        filename(PathLike): Path to bounding box file.

    Returns:
        format(str): Inferred format of the bounding box file.
    """
    filename = str(filename)
    if is_mot(filename):
        return 'mot'
    if filename.endswith('.csv'):
        return 'soccertrack_bbox'
    if filename.endswith('.ndjson'):
        return 'labelbox'
    raise ValueError('Could not infer file format')
