def infer_gps_format(filename: PathLike) ->str:
    """Try to infer GPS format from filename.

    Args:
        filename (Union[str, bytes, os.PathLike[Any]]): Filename to infer format from.
    """
    filename = str(filename)
    if is_soccertrack_coordinates(filename):
        return 'soccertrack'
    if filename.endswith('.xlsx'):
        return 'gpsports'
    if filename.endswith('.csv'):
        return 'statsports'
    raise ValueError('Could not infer file format')
