def rename_key(key):
    regex = '\\w+[.]\\d+'
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, '_'.join(pat.split('.')))
    return key
