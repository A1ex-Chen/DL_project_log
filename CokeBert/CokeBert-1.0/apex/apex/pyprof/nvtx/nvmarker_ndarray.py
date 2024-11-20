def ndarray(arg, name=''):
    a = {}
    a['name'] = name
    a['type'] = 'ndarray'
    a['shape'] = arg.shape
    a['dtype'] = str(arg.dtype).split('.')[-1]
    cadena['args'].append(a)
