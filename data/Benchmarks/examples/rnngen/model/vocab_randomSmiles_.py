def randomSmiles_(m1):
    m1.SetProp('_canonicalRankingNumbers', 'True')
    idxs = list(range(0, m1.GetNumAtoms()))
    random.shuffle(idxs)
    for i, v in enumerate(idxs):
        m1.GetAtomWithIdx(i).SetProp('_canonicalRankingNumber', str(v))
    return Chem.MolToSmiles(m1)
