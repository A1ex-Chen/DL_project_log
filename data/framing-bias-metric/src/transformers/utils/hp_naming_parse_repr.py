@classmethod
def parse_repr(cls, repr):
    repr = repr[len(cls.PREFIX) + 1:]
    if repr == '':
        values = []
    else:
        values = repr.split('_')
    parameters = {}
    for value in values:
        if '-' in value:
            p_k, p_v = value.split('-')
        else:
            p_k = re.sub('[0-9.]', '', value)
            p_v = float(re.sub('[^0-9.]', '', value))
        key = cls.NAMING_INFO['reverse_short_param'][p_k]
        parameters[key] = p_v
    for k in cls.DEFAULTS:
        if k not in parameters:
            parameters[k] = cls.DEFAULTS[k]
    return parameters
