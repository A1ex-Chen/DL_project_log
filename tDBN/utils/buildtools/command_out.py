def out(path):
    return Path(path).parent / (Path(path).stem + '.o')
