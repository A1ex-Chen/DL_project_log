def _parse_readme(lns):
    """Get link and metadata from opus model card equivalent."""
    subres = {}
    for ln in [x.strip() for x in lns]:
        if not ln.startswith('*'):
            continue
        ln = ln[1:].strip()
        for k in ['download', 'dataset', 'models', 'model', 'pre-processing']:
            if ln.startswith(k):
                break
        else:
            continue
        if k in ['dataset', 'model', 'pre-processing']:
            splat = ln.split(':')
            _, v = splat
            subres[k] = v
        elif k == 'download':
            v = ln.split('(')[-1][:-1]
            subres[k] = v
    return subres
