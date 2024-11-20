def infer_metadata_from_filename(filename: PathLike) ->Mapping[str, int]:
    """Try to infer metadata from filename.

    Args:
        filename (Union[str, bytes, os.PathLike[Any]]): Filename to infer metadata from

    Returns:
        dict[str, Union[int, str]]: Dictionary with metadata
    """
    filename = Path(filename)
    basename = filename.name
    try:
        teamid = int(basename.split('_')[1])
        playerid = int(basename.split('_')[2].split('.')[0])
    except (IndexError, ValueError):
        teamid = 0
        playerid = 0
    metadata = {'teamid': teamid, 'playerid': playerid}
    return metadata
