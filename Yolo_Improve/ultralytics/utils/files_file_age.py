def file_age(path=__file__):
    """Return days since last file update."""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)
    return dt.days
