def load_soccertrack_coordinates(filename: PathLike, playerid: (int | None)
    =None, teamid: (int | None)=None) ->CoordinatesDataFrame:
    """Load CoordinatesDataFrame from soccertrack coordinates file.

    Args:
        filename(str): Path to soccertrack coordinates file.

    Returns:
        soccertrack_coordinates_dataframe(CoordinatesDataFrame): DataFrame of soccertrack coordinates file.
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
    return df
