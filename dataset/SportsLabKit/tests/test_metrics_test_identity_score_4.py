def test_identity_score_4(self):
    """Test for IDENTITY Score when an object is missing in the middle."""
    bboxes_track = player_dfs[0].copy()
    bboxes_track.loc[1] = -1
    bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)
    identity = identity_score(bboxes_track, bboxes_gt)
    IDTP = 1
    IDFN = 3
    IDFP = 0
    IDR = IDTP / (IDTP + IDFN)
    IDP = IDTP / (IDTP + IDFP)
    IDF1 = 2 * (IDR * IDP) / (IDR + IDP)
    ans = {'IDF1': IDF1, 'IDR': IDR, 'IDP': IDP, 'IDTP': IDTP, 'IDFN': IDFN,
        'IDFP': IDFP}
    self.assertDictEqual(identity, ans)
