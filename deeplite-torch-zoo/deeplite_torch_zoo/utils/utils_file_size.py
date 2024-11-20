def file_size(path):
    mb = 1 << 20
    path = Path(path)
    if path.is_file():
        val = path.stat().st_size / mb
    elif path.is_dir():
        val = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()
            ) / mb
    else:
        val = 0.0
    return val
