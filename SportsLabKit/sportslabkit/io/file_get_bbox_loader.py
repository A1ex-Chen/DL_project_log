def get_bbox_loader(format: str) ->Callable[[PathLike], BBoxDataFrame]:
    """Returns a function that loads the corresponding bbox format.

    Args:
        format(str): bbox format to load.

    Returns:
        bbox_loader(Callable[[PathLike], BBoxDataFrame]): Function that loads the corresponding bbox format.
    """
    format = format.lower()
    if format == 'mot':
        return load_mot
    if format == 'labelbox':
        return load_labelbox
    if format == 'soccertrack_bbox':
        return load_soccertrack_bbox
    raise ValueError(f'Unknown format {format}')
