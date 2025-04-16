def is_writeable(dir, test=False):
    if not test:
        return os.access(dir, os.W_OK)
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):
            pass
        file.unlink()
        return True
    except OSError:
        return False
