def is_ascii(s=''):
    s = str(s)
    return len(s.encode().decode('ascii', 'ignore')) == len(s)
