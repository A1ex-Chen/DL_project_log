def _get_regular_import_name(path, module_paths):
    path = Path(path)
    for mp in module_paths:
        mp = Path(mp)
        if mp == path:
            return path.stem
        try:
            relative_path = path.relative_to(Path(mp))
            parts = list((relative_path.parent / relative_path.stem).parts)
            module_name = '.'.join([mp.stem] + parts)
            return module_name
        except:
            pass
    return None
