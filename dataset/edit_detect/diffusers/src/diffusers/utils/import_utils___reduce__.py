def __reduce__(self):
    return self.__class__, (self._name, self.__file__, self._import_structure)
