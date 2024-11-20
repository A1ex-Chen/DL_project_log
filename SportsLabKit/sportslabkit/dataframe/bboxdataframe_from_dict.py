@staticmethod
def from_dict(d: dict, attributes: (Iterable[str] | None)=('bb_left',
    'bb_top', 'bb_width', 'bb_height')):
    """Create a BBoxDataFrame from a nested dictionary contating the coordinates of the players and the ball.

        The input dictionary should be of the form:
        {
            home_team_key: {
                PlayerID: {frame: [x, y], ...},
                PlayerID: {frame: [x, y], ...},
                ...
            },
            away_team_key: {
                PlayerID: {frame: [x, y], ...},
                PlayerID: {frame: [x, y], ...},
                ...
            },
            ball_key: {
                frame: [x, y],
                frame: [x, y],
                ...
            }
        }
        The `PlayerID` can be any unique identifier for the player, e.g. their jersey number or name. The PlayerID for the ball can be omitted, as it will be set to "0". `frame` must be an integer identifier for the frame number.

        Args:
            dict (dict): Nested dictionary containing the coordinates of the players and the ball.
            attributes (Optional[Iterable[str]], optional): Attributes to use for the coordinates. Defaults to ("x", "y").

        Returns:
            CoordinatesDataFrame: CoordinatesDataFrame.
        """
    attributes = list(attributes)
    data = []
    for team, team_dict in d.items():
        for player, player_dict in team_dict.items():
            for frame, bbox in player_dict.items():
                data.append([team, player, frame, *bbox])
    df = pd.DataFrame(data, columns=['TeamID', 'PlayerID', 'frame', *
        attributes])
    df.pivot(index='frame', columns=['TeamID', 'PlayerID'], values=attributes)
    df = df.pivot(index='frame', columns=['TeamID', 'PlayerID'], values=
        attributes)
    multi_index = pd.MultiIndex.from_tuples(df.columns.swaplevel(0, 1).
        swaplevel(1, 2))
    df.columns = pd.MultiIndex.from_tuples(multi_index)
    df.rename_axis(['TeamID', 'PlayerID', 'Attributes'], axis=1, inplace=True)
    df.sort_index(axis=1, inplace=True)
    return BBoxDataFrame(df)
