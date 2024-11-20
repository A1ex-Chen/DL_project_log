@staticmethod
def from_numpy(arr: np.ndarray, team_ids: (Iterable[str] | None)=None,
    player_ids: (Iterable[int] | None)=None, attributes: (Iterable[str] |
    None)=('x', 'y'), auto_fix_columns: bool=True):
    """Create a CoordinatesDataFrame from a numpy array of either shape (L, N, 2) or (L, N * 2) where L is the number of frames, N is the number of players and 2 is the number of coordinates (x, y).

        Args:
            arr : Numpy array.
            team_ids : Team ids. Defaults to None. If None, team ids will be set to 0 for all players. If not None, must have the same length as player_ids
            Player ids: Player ids. Defaults to None. If None, player ids will be set to 0 for all players. If not None, must have the same length as team_ids
            attributes : Attribute names to use. Defaults to ("x", "y").
            auto_fix_columns : If True, will automatically fix the team_ids, player_ids and attributes so that they are equal to the number of columns. Defaults to True.


        Returns:
            CoordinatesDataFrame: CoordinatesDataFrame.

        Examples:
            >>> from soccertrack.dataframe import CoordinatesDataFrame
            >>> import numpy as np
            >>> arr = np.random.rand(10, 22, 2)
            >>> codf = CoordinatesDataFrame.from_numpy(arr, team_ids=["0"] * 22, player_ids=list(range(22)))

        """
    n_frames, n_players, n_attributes = arr.shape
    n_columns = n_players * n_attributes
    if team_ids and player_ids:
        assert len(team_ids) == len(player_ids
            ), f'team_ids and player_ids must have the same length. Got {len(team_ids)} and {len(player_ids)} respectively.'
    assert arr.ndim in (2, 3), 'Array must be of shape (L, N, 2) or (L, N * 2)'
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    df = pd.DataFrame(arr)
    if team_ids is None:
        if n_players == 23:
            team_ids = ['0'] * 22 + ['1'] * 22 + ['ball'] * 2
        else:
            team_ids = ['0'] * n_players * n_attributes
    elif auto_fix_columns and len(team_ids) != n_columns:
        team_ids = np.repeat(team_ids, n_attributes)
    if player_ids is None:
        if n_players == 23:
            _players = list(np.linspace(0, 10, 22).round().astype(int))
            player_ids = _players + _players + [0, 0]
        else:
            player_ids = list(range(n_players)) * n_attributes
    elif auto_fix_columns and len(player_ids) != player_ids:
        player_ids = np.repeat(player_ids, n_attributes)
    attributes = attributes * n_players

    def _assert_correct_length(x, key):
        assert len(x
            ) == n_columns, f'Incorrect number of resulting {key} columns: {len(x)} != {n_columns}. Set auto_fix_columns to False to disable automatic fixing of columns. See docs for more information.'
    _assert_correct_length(team_ids, 'TeamID')
    _assert_correct_length(player_ids, 'PlayerID')
    _assert_correct_length(attributes, 'Attributes')
    idx = pd.MultiIndex.from_arrays([team_ids, player_ids, attributes])
    df = CoordinatesDataFrame(df.values, index=df.index, columns=idx)
    df.rename_axis(['TeamID', 'PlayerID', 'Attributes'], axis=1, inplace=True)
    df.index.name = 'frame'
    return CoordinatesDataFrame(df)
