def load_soccertrack_bbox(filename: PathLike) ->pd.DataFrame:
    """Load a dataframe from a file.

    Args:
        filename (PathLike): Path to load the dataframe.
    Returns:
        df (pd.DataFrame): Dataframe loaded from the file.
    """
    attrs = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                k, v = line[1:].strip().split(':')
                attrs[k] = auto_string_parser(v)
            else:
                break
    skiprows = len(attrs)
    df = pd.read_csv(filename, header=[0, 1, 2], index_col=0, skiprows=skiprows
        )
    df.attrs = attrs
    id_list = []
    for column in df.columns:
        team_id = column[0]
        player_id = column[1]
        id_list.append((team_id, player_id))
    for id in sorted(set(id_list)):
        df.loc[:, (id[0], id[1], 'conf')] = 1.0
    df = df[df.sort_index(axis=1, level=[0, 1], ascending=[True, True]).columns
        ]
    return df
