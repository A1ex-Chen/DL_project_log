def tensor(arg, name=''):
    a = {}
    a['name'] = name
    a['type'] = 'tensor'
    a['shape'] = tuple(arg.size())
    a['dtype'] = str(arg.dtype).split('.')[-1]
    cadena['args'].append(a)
