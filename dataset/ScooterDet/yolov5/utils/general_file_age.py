def file_age(path=__file__):
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)
    return dt.days
