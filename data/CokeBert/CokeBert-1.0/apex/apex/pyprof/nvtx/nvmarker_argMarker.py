def argMarker(mod, op, args, kwargs):

    def tensor(arg, name=''):
        a = {}
        a['name'] = name
        a['type'] = 'tensor'
        a['shape'] = tuple(arg.size())
        a['dtype'] = str(arg.dtype).split('.')[-1]
        cadena['args'].append(a)

    def ndarray(arg, name=''):
        a = {}
        a['name'] = name
        a['type'] = 'ndarray'
        a['shape'] = arg.shape
        a['dtype'] = str(arg.dtype).split('.')[-1]
        cadena['args'].append(a)

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

    def isscalar(arg):
        return type(arg) is int or type(arg) is float or type(arg
            ) is bool or arg is None or type(arg) is str

    def issequence(arg):
        return isinstance(arg, list) or isinstance(arg, tuple)

    def foo(args, name):
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.dim() == 0:
                    scalar(arg.item(), name)
                else:
                    tensor(arg, name)
            elif isinstance(arg, numpy.ndarray):
                ndarray(arg, name)
            elif isscalar(arg):
                scalar(arg, name)
            elif issequence(arg):
                if len(arg) == 0 or isscalar(arg[0]):
                    seq(arg, name)
                else:
                    foo(arg, name)
            """
			else:
				print("The following arg is none of Tensor, numpy array, scalar but a %s" % (str(type(arg))))
				print("Mod: %s" % str(mod.__name__))
				print("Op: %s" % str(op))
				print(dir(arg))
			"""
    cadena = {}
    cadena['mod'] = mod.__name__
    cadena['op'] = op
    cadena['args'] = []
    foo(args, '')
    for k, v in kwargs.items():
        foo((v,), k)
    return str(cadena)
