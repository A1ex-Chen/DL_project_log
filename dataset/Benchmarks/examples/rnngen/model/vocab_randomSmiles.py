def randomSmiles(smi, max_len=150, attempts=100):
    import random
    from rdkit import Chem

    def randomSmiles_(m1):
        m1.SetProp('_canonicalRankingNumbers', 'True')
        idxs = list(range(0, m1.GetNumAtoms()))
        random.shuffle(idxs)
        for i, v in enumerate(idxs):
            m1.GetAtomWithIdx(i).SetProp('_canonicalRankingNumber', str(v))
        return Chem.MolToSmiles(m1)
    m1 = Chem.MolFromSmiles(smi)
    if m1 is None:
        return None
    if m1 is not None and attempts == 1:
        return [smi]
    s = set()
    for i in range(attempts):
        smiles = randomSmiles_(m1)
        s.add(smiles)
    s = list(filter(lambda x: len(x) < max_len, list(s)))
    if len(s) > 1:
        return s
    else:
        return [smi]
