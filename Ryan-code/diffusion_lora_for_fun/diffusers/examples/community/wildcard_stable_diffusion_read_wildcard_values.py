def read_wildcard_values(path: str):
    with open(path, encoding='utf8') as f:
        return f.read().splitlines()
