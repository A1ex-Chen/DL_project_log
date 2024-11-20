def test_identity_score_1(self):
    """Test IDENTITY score with perfect detection."""
    bboxes_track = bbdf
    bboxes_gt = bbdf
    identity = identity_score(bboxes_track, bboxes_gt)
    IDTP = 46
    IDFN = 0
    IDFP = 0
    IDR = IDTP / (IDTP + IDFN)
    IDP = IDTP / (IDTP + IDFP)
    IDF1 = 2 * (IDR * IDP) / (IDR + IDP)
    ans = {'IDF1': IDF1, 'IDR': IDR, 'IDP': IDP, 'IDTP': IDTP, 'IDFN': IDFN,
        'IDFP': IDFP}
    self.assertDictEqual(identity, ans)
