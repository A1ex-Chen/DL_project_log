def toMol(smi):
    try:
        m = pybel.readstring('smi', smi)
        return m.write('smi')
    except Exception:
        return None
