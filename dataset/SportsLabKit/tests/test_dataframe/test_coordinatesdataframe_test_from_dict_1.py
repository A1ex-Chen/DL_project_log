def test_from_dict_1(self):
    d = {'home_team': {'player_1a': {(1): (1, 2), (2): (3, 4), (3): (5, 6)},
        'player_1b': {(1): (7, 8), (2): (9, 10), (3): (11, 12)}},
        'away_team': {'player_2a': {(1): (13, 14), (2): (15, 16), (3): (17,
        18)}, 'player_2b': {(1): (19, 20), (2): (21, 22), (3): (23, 24)}},
        'ball': {'ball': {(1): (25, 26), (2): (27, 28), (3): (29, 30)}}}
    codf = CoordinatesDataFrame.from_dict(d)
    assert codf.shape == (3, 10)
