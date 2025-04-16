def test_preprocess_for_mot_eval_2(self):
    """Test for when there are missing frames"""
    bbdf = BBoxDataFrame.from_dict({'home': {'1': {(0): [10, 10, 25, 25, 1],
        (2): [5, 0, 25, 25, 1]}}}, attributes=['bb_left', 'bb_top',
        'bb_width', 'bb_height', 'conf'])
    ids, dets = bbdf.preprocess_for_mot_eval()
    ans_ids = [np.array([0]), np.array([]), np.array([0])]
    ans_dets = [[np.array([10, 10, 25, 25])], [], [np.array([5, 0, 25, 25])]]
    for i in range(len(ids)):
        np.testing.assert_almost_equal(ids[i], ans_ids[i])
    for i in range(len(dets)):
        np.testing.assert_almost_equal(dets[i], ans_dets[i])
