def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if '.' in att:
        obj = getattr_recursive(obj, '.'.join(att.split('.')[:-1]))
    setattr(obj, att.split('.')[-1], val)
