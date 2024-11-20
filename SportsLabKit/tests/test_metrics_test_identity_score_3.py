def test_identity_score_3(self):
    """Test IDENTITY score with half tracking."""
    bboxes_track = player_dfs[0]
    bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)
    identity = identity_score(bboxes_track, bboxes_gt)
    IDTP = 2
    IDFN = 2
    IDFP = 0
    IDR = IDTP / (IDTP + IDFN)
    IDP = IDTP / (IDTP + IDFP)
    IDF1 = 2 * (IDR * IDP) / (IDR + IDP)
    ans = {'IDF1': IDF1, 'IDR': IDR, 'IDP': IDP, 'IDTP': IDTP, 'IDFN': IDFN,
        'IDFP': IDFP}
    self.assertDictEqual(identity, ans)
