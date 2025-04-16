def test_test_preprocess_for_mot_eval_3(self):
    """Test for bug that occured when the frames did not start at 0 and has nan value frames."""
    bbdf = BBoxDataFrame.from_dict({'home': {'1': {(2): [10, 10, 25, 25, 1],
        (3): [0, 0, 20, 20, 1]}}}, attributes=['bb_left', 'bb_top',
        'bb_width', 'bb_height', 'conf'])
    bbdf = bbdf.reindex(range(1, 4))
    ids, dets = bbdf.preprocess_for_mot_eval()
    ans_ids = [[], np.array([0]), np.array([0])]
    ans_dets = [[], [np.array([10, 10, 25, 25])], [np.array([0, 0, 20, 20])]]
    for i in range(len(ids)):
        np.testing.assert_almost_equal(ids[i], ans_ids[i])
    for i in range(len(dets)):
        np.testing.assert_almost_equal(dets[i], ans_dets[i])
