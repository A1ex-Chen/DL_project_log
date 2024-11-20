def load_codf(filename: PathLike, format: (str | None)=None, playerid: (int |
    None)=None, teamid: (int | None)=None) ->CoordinatesDataFrame:
    """Load CoordinatesDataFrame from file.

    Args:
        filename (Union[str, bytes, os.PathLike[Any]]): Filename to load from.
        format (Optional[str], optional): Format of GPS data. Defaults to None.
        playerid (Optional[int], optional): Player ID. Defaults to None.
        teamid (Optional[int], optional): Team ID. Defaults to None.

    Raises:
        ValueError: If format is not provided and could not be inferred.

    Returns:
        CoordinatesDataFrame: DataFrame of GPS data.
    """
    if format is None:
        format = infer_gps_format(filename)
    loader = get_gps_loader(format)
    df = CoordinatesDataFrame(loader(filename))
    df.rename_axis(['TeamID', 'PlayerID', 'Attributes'], axis=1, inplace=True)
    return df
