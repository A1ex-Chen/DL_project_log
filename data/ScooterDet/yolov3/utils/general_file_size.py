def file_size(path):
    mb = 1 << 20
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()
            ) / mb
    else:
        return 0.0
