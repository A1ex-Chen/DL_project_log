def eval(self, *args, **kw):
    for p in self.prev:
        if not p.eval(*args, **kw):
            self.state = NodeState.Error
            return False
    if self.state == NodeState.Normal:
        if self._eval(*args, **kw):
            self.state = NodeState.Evaled
        else:
            self.state = NodeState.Error
            return True
    return True
