def compute_MR(origin, target):
    mct, total = 0, 0
    res = {}
    for o, t in zip(origin, target):
        main_idx = str(int(o['id']) % int(1000000.0))
        if main_idx in res:
            continue
        oid = o['options'].index(o['answer'])
        res[main_idx] = 1
        total += 1
        assert o['id'] == t['id']
        tid = t['options'].index(t['answer'])
        if oid == tid:
            mct += 1
    return mct / total * 100
