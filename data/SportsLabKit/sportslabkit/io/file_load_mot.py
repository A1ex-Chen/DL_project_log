def load_mot(filename: PathLike) ->CoordinatesDataFrame:
    """Load MOT format file to CoordinatesDataFrame.

    Args:
        filename(str): Path to statsports file.

    Returns:
        statsports_dataframe(CoordinatesDataFrame): DataFrame of statsports file.

    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(STATSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    groups = pd.read_csv(filename, usecols=[0, 1, 2, 3, 4, 5], index_col=0
        ).groupby('id')
    teamid = 0
    df_list = []
    for playerid, group in groups:
        group['conf'] = 1.0
        group['class_id'] = int(0)
        if playerid == 23:
            group['class_id'] = int(32)
            teamid = 3
            playerid = 0
        elif 11 < playerid < 23:
            teamid = 1
            playerid = playerid - 11
        bbox_cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']
        idx = pd.MultiIndex.from_arrays([[int(teamid)] * 5, [int(playerid)] *
            5, bbox_cols])
        bbox_df = BBoxDataFrame(group[bbox_cols].values, index=group.index,
            columns=idx)
        df_list.append(bbox_df)
    merged_dataframe = df_list[0].join(df_list[1:len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()
    merged_dataframe = df_list[0].join(df_list[1:len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()
    return merged_dataframe
