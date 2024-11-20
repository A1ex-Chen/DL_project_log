def test_identity_score_2(self):
    """Test IDENTITY score with zero tracking."""
    bboxes_track = bbdf[0:2].iloc[0:0]
    bboxes_gt = bbdf
    identity = identity_score(bboxes_track, bboxes_gt)
    IDTP = 0
    IDFN = 46
    IDFP = 0
    IDR = IDTP / (IDTP + IDFN)
    IDP = 0
    IDF1 = 0
    ans = {'IDF1': IDF1, 'IDR': IDR, 'IDP': IDP, 'IDTP': IDTP, 'IDFN': IDFN,
        'IDFP': IDFP}
    self.assertDictEqual(identity, ans)
