def count_valid_samples(smiles, rdkit=True):
    if rdkit:
        from rdkit import Chem, RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        def toMol(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(mol)
            except Exception:
                return None
    else:
        import pybel

        def toMol(smi):
            try:
                m = pybel.readstring('smi', smi)
                return m.write('smi')
            except Exception:
                return None
    count = 0
    goods = []
    for smi in smiles:
        try:
            mol = toMol(smi)
            if mol is not None:
                goods.append(mol)
                count += 1
        except Exception:
            continue
    return count, goods
