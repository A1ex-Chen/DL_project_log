def __init__(self, name, build_func=None, parent=None, scope=None):
    self._name = name
    self._module_dict = dict()
    self._children = dict()
    self._scope = self.infer_scope() if scope is None else scope
    if build_func is None:
        if parent is not None:
            self.build_func = parent.build_func
        else:
            self.build_func = build_from_cfg
    else:
        self.build_func = build_func
    if parent is not None:
        assert isinstance(parent, Registry)
        parent._add_children(self)
        self.parent = parent
    else:
        self.parent = None
