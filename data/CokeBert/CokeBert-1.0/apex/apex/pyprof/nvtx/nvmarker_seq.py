def seq(arg, name=''):
    assert issequence(arg)
    a = {}
    a['name'] = name
    if isinstance(arg, list):
        a['type'] = 'list'
        a['value'] = arg
    else:
        a['type'] = 'tuple'
        a['value'] = tuple(arg)
    cadena['args'].append(a)
