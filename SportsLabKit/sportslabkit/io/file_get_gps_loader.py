def get_gps_loader(format: str) ->Callable[[PathLike, int, int],
    CoordinatesDataFrame]:
    """Get GPS loader function for a given format.

    Args:
        format (str): GPS format.

    Returns:
        Callable[[PathLike, int, int], CoordinatesDataFrame]: GPS loader function.
    """
    format = format.lower()
    if format == 'gpsports':
        return load_gpsports
    if format == 'statsports':
        return load_statsports
    if format == 'soccertrack':
        return load_soccertrack_coordinates
    raise ValueError(f'Unknown format {format}')
