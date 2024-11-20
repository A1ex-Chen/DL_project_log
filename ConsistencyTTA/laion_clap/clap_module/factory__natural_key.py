def _natural_key(string_):
    return [(int(s) if s.isdigit() else s) for s in re.split('(\\d+)',
        string_.lower())]
