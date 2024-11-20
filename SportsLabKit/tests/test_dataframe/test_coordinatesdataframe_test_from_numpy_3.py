def test_from_numpy_3(self):
    arr = np.random.rand(5, 3, 2)
    team_ids = [1, 2, 'ball']
    player_ids = [1, 1, 'ball']
    codf = CoordinatesDataFrame.from_numpy(arr, team_ids, player_ids)
    assert codf.shape == (5, 6)
