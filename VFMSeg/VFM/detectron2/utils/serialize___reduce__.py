def __reduce__(self):
    s = cloudpickle.dumps(self._obj)
    return cloudpickle.loads, (s,)
