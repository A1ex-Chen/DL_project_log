def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)
    if path.exists() and exist_ok or not path.exists():
        return str(path)
    else:
        dirs = glob.glob(f'{path}{sep}*')
        matches = [re.search(f'%s{sep}(\\d+)' % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f'{path}{sep}{n}'
