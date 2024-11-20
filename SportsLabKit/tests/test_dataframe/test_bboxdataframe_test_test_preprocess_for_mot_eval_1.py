def test_test_preprocess_for_mot_eval_1(self):
    """Test for when there are no missing frames"""
    bbdf = BBoxDataFrame.from_dict({'home': {'1': {(0): [10, 10, 25, 25, 1],
        (1): [0, 0, 20, 20, 1]}, '2': {(2): [2, 1, 25, 25, 1]}}},
        attributes=['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    ids, dets = bbdf.preprocess_for_mot_eval()
    ans_ids = [np.array([0]), np.array([0]), np.array([1])]
    ans_dets = [[np.array([10, 10, 25, 25])], [np.array([0, 0, 20, 20])], [
        np.array([2, 1, 25, 25])]]
    for i in range(len(ids)):
        np.testing.assert_almost_equal(ids[i], ans_ids[i])
    for i in range(len(dets)):
        np.testing.assert_almost_equal(dets[i], ans_dets[i])
