def isfunc(mod, f):
    assert hasattr(mod, f)
    attr = getattr(mod, f)
    if len(f) >= 2:
        if f[0] == '_' and f[1] != '_':
            return False
    ignore = ['__all__', '__array__', '__array_priority__',
        '__array_wrap__', '__bool__', '__builtins__', '__cached__',
        '__class__', '__deepcopy__', '__delattr__', '__delitem__',
        '__dict__', '__dir__', '__doc__', '__file__', '__format__',
        '__getattribute__', '__getitem__', '__hash__', '__index__',
        '__init__', '__init_subclass__', '__iter__', '__len__',
        '__loader__', '__module__', '__name__', '__new__', '__nonzero__',
        '__package__', '__path__', '__reduce__', '__reduce_ex__',
        '__repr__', '__reversed__', '__setattr__', '__setitem__',
        '__setstate__', '__sizeof__', '__spec__', '__str__',
        '__subclasshook__', '__version__', '__weakref__']
    ignore += ['size', 'tolist', 'dim', 'is_storage', 'item']
    if f in ignore:
        return False
    return ins.ismethod(attr) or ins.isfunction(attr
        ) or ins.ismethoddescriptor(attr) or ins.isbuiltin(attr)
