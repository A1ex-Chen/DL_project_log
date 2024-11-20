def __init__(self, name, module_file, import_structure, module_spec=None,
    extra_objects=None):
    super().__init__(name)
    self._modules = set(import_structure.keys())
    self._class_to_module = {}
    for key, values in import_structure.items():
        for value in values:
            self._class_to_module[value] = key
    self.__all__ = list(import_structure.keys()) + list(chain(*
        import_structure.values()))
    self.__file__ = module_file
    self.__spec__ = module_spec
    self.__path__ = [os.path.dirname(module_file)]
    self._objects = {} if extra_objects is None else extra_objects
    self._name = name
    self._import_structure = import_structure
