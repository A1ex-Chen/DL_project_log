def load_labelbox(filename: PathLike) ->CoordinatesDataFrame:
    """Load labelbox format file to CoordinatesDataFrame.

    Args:
        filename(str): Path to gpsports file.

    Returns:
        gpsports_dataframe(CoordinatesDataFrame): DataFrame of gpsports file.

    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(GPSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    df = pd.read_json(filename, lines=True).explode('objects')
    objects_df = df['objects'].apply(pd.Series)
    bbox_df = objects_df['bbox'].apply(pd.Series)
    df = pd.concat([df[['frameNumber']], objects_df.title.str.split('_',
        expand=True), bbox_df[['left', 'top', 'width', 'height']]], axis=1)
    df.columns = ['frame', 'teamid', 'playerid', 'bb_left', 'bb_top',
        'bb_width', 'bb_height']
    df.set_index('frame', inplace=True)
    groups = df.groupby('playerid', dropna=False)
    df_list = []
    for playerid, group in groups:
        teamid = group.teamid.iloc[0]
        bbox_cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height']
        if teamid.lower() == 'ball':
            teamid = 3
            playerid = 0
        idx = pd.MultiIndex.from_arrays([[int(float(teamid))] * 4, [int(
            float(playerid))] * 4, bbox_cols])
        bbox_df = BBoxDataFrame(group[bbox_cols].values, index=group.index,
            columns=idx)
        df_list.append(bbox_df)
    merged_dataframe = df_list[0].join(df_list[1:len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()
    return merged_dataframe
