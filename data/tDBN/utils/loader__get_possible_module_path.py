def _get_possible_module_path(paths):
    ret = []
    for p in paths:
        p = Path(p)
        for path in p.glob('*'):
            if path.suffix in ['py', '.so'] or path.is_dir():
                if path.stem.isidentifier():
                    ret.append(path)
    return ret
