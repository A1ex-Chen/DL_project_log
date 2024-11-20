def getSeqId(mlist):
    """
			Get sequence ids from seq / alt seq marker list.
			"""
    ids = []
    assert type(mlist) == list
    for m in mlist:
        assert ', seq = ' in m
        seq = int(m.split('=')[1])
        ids.append(seq)
    ids = list(set(ids))
    ids.sort()
    return ids
