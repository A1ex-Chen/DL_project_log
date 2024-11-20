def load_gpsports(filename: PathLike, playerid: (int | None)=None, teamid:
    (int | None)=None) ->CoordinatesDataFrame:
    """Load CoordinatesDataFrame from GPSPORTS file.

    Args:
        filename(str): Path to gpsports file.

    Returns:
        gpsports_dataframe(CoordinatesDataFrame): DataFrame of gpsports file.
    """
    raw_df = pd.read_excel(filename, skiprows=7, usecols=['Time',
        'Latitude', 'Longitude'], index_col='Time').rename(columns={
        'Latitude': 'Lat', 'Longitude': 'Lon'})
    metadata = infer_metadata_from_filename(filename)
    teamid = teamid if teamid is not None else metadata['teamid']
    playerid = playerid if playerid is not None else metadata['playerid']
    idx = pd.MultiIndex.from_arrays([[int(teamid)] * 2, [int(playerid)] * 2,
        list(raw_df.columns)])
    gpsports_dataframe = CoordinatesDataFrame(raw_df.values, index=raw_df.
        index, columns=idx)
    gpsports_dataframe.index = gpsports_dataframe.index.map(lambda x: x.time())
    return gpsports_dataframe
