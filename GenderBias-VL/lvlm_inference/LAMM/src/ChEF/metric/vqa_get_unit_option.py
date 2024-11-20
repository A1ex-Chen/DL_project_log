def get_unit_option(splits, choices='ABCD', prefix='', suffix=''):
    res = []
    for c in choices:
        if prefix + c + suffix in splits:
            res.append(c)
    return res
