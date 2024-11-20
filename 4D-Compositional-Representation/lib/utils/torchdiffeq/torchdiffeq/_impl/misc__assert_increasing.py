def _assert_increasing(t):
    assert (t[1:] > t[:-1]).all(), 't must be strictly increasing or decrasing'
