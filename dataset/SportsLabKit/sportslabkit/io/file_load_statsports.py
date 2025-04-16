def load_statsports(filename: PathLike, playerid: (int | None)=None, teamid:
    (int | None)=None) ->CoordinatesDataFrame:
    """Load CoordinatesDataFrame from STATSPORTS file.

    Args:
        filename(str): Path to statsports file.

    Returns:
        statsports_dataframe(CoordinatesDataFrame): DataFrame of statsports file.
    """
    raw_df = pd.read_csv(filename).iloc[:, [1, 3, 4]].set_axis(['Time',
        'Lat', 'Lon'], axis='columns').reset_index(drop=True)
    raw_df['Time'] = pd.to_datetime(raw_df['Time'])
    raw_df.set_index('Time', inplace=True)
    metadata = infer_metadata_from_filename(filename)
    teamid = teamid if teamid is not None else metadata['teamid']
    playerid = playerid if playerid is not None else metadata['playerid']
    idx = pd.MultiIndex.from_arrays([[int(teamid)] * 2, [int(playerid)] * 2,
        list(raw_df.columns)])
    statsports_dataframe = CoordinatesDataFrame(raw_df.values, index=raw_df
        .index, columns=idx)
    statsports_dataframe.index = statsports_dataframe.index.map(lambda x: x
        .time())
    return statsports_dataframe
