def load_gps(filenames: (Sequence[PathLike,] | PathLike), playerids: (
    Sequence[int] | int)=(), teamids: (Sequence[int] | int)=()
    ) ->CoordinatesDataFrame:
    """Load GPS data from multiple files.

    Args:
        gpsports_dataframe(CoordinatesDataFrame): DataFrame of gpsports file.
        statsports_dataframe(CoordinatesDataFrame): DataFrame of statsports file.

    Returns:
        merged_dataframe(CoordinatesDataFrame): DataFrame of merged gpsports and statsports.
    """
    if not isinstance(filenames, Sequence):
        filenames = [filenames]
    playerid = 0
    teamid = None
    if not isinstance(playerids, Sequence):
        playerids = [playerids]
    if not isinstance(teamids, Sequence):
        teamids = [teamids]
    df_list = []
    for i, (filename, playerid, teamid) in enumerate(zip_longest(filenames,
        playerids, teamids)):
        playerid = playerid if playerid is not None else i
        gps_format = infer_gps_format(filename)
        dataframe = get_gps_loader(gps_format)(filename, playerid, teamid)
        df_list.append(dataframe)
        playerid += 1
    merged_dataframe = df_list[0].join(df_list[1:len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()
    merged_dataframe = df_list[0].join(df_list[1:len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()
    return merged_dataframe
