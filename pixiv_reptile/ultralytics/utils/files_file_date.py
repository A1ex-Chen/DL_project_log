def file_date(path=__file__):
    """Return human-readable file modification date, i.e. '2021-3-26'."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'
