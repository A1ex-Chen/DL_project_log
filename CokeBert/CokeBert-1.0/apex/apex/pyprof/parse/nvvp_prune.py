def prune(mlist):
    """
			Remove markers with the same seqId and if the strings are similar.
			This function works on a sorted sequence.
			"""
    assert type(mlist) == list
    assert len(mlist)
    a = mlist[0:1]
    for i in range(1, len(mlist)):
        m = mlist[i]
        pm = mlist[i - 1]
        name, seq = m.split(',')
        pname, pseq = pm.split(',')
        similar = name in pname or pname in name
        if seq == pseq and similar:
            continue
        else:
            a.append(m)
    return a
