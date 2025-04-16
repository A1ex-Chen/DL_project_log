def test_to_mot_eval_format_1(self):
    gt_bbdf = BBoxDataFrame.from_dict({'home': {'1': {(0): [10, 10, 25, 25,
        1], (1): [0, 0, 25, 25, 1], (2): [5, 0, 25, 25, 1]}, '2': {(2): [0,
        5, 25, 25, 1]}}}, attributes=['bb_left', 'bb_top', 'bb_width',
        'bb_height', 'conf'])
    pred_bbdf = BBoxDataFrame.from_dict({'home': {'1': {(0): [10, 10, 25, 
        25, 1], (1): [0, 0, 20, 20, 1]}, '2': {(2): [2, 1, 25, 25, 1]}}},
        attributes=['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    data = to_mot_eval_format(gt_bbdf, pred_bbdf)
    ans = {'tracker_ids': [[0], [0], [1]], 'gt_ids': [[0], [0], [0, 1]],
        'tracker_dets': [[[10, 10, 25, 25]], [[0, 0, 20, 20]], [[2, 1, 25, 
        25]]], 'gt_dets': [[[10, 10, 25, 25]], [[0, 0, 25, 25]], [[5, 0, 25,
        25], [0, 5, 25, 25]]], 'similarity_scores': [np.array([[1.0]]), np.
        array([[0.64]]), np.array([[0.73130194], [0.62972621]])],
        'num_tracker_dets': 3, 'num_gt_dets': 4, 'num_tracker_ids': 2,
        'num_gt_ids': 2, 'num_timesteps': 3}
    for key in ans.keys():
        d = data[key]
        a = ans[key]
        if key in ['similarity_scores', 'tracker_dets', 'gt_dets',
            'tracker_ids', 'gt_ids']:
            for i in range(len(d)):
                np.testing.assert_allclose(d[i], a[i])
        else:
            self.assertEqual(d, a)
