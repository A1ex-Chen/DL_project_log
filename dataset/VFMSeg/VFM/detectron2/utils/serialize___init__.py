def __init__(self, obj):
    while isinstance(obj, PicklableWrapper):
        obj = obj._obj
    self._obj = obj
