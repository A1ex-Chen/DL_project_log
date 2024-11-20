def scalar(arg, name=''):
    a = {}
    a['name'] = name
    a['type'] = type(arg).__name__
    if arg == float('inf'):
        a['value'] = 'inf'
    elif arg == float('-inf'):
        a['value'] = '-inf'
    elif isinstance(arg, float) and math.isnan(arg):
        a['value'] = 'nan'
    else:
        a['value'] = arg
    cadena['args'].append(a)
