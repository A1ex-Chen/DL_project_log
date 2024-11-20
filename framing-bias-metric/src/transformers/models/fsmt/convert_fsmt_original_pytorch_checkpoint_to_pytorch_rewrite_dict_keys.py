def rewrite_dict_keys(d):
    d2 = dict((re.sub('@@$', '', k), v) if k.endswith('@@') else (re.sub(
        '$', '</w>', k), v) for k, v in d.items())
    keep_keys = '<s> <pad> </s> <unk>'.split()
    for k in keep_keys:
        del d2[f'{k}</w>']
        d2[k] = d[k]
    return d2
