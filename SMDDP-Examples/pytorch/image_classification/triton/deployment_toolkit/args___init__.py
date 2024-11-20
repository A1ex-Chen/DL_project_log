def __init__(self, cls_or_fn, module_path: Optional[str]=None):
    self._cls_or_fn = cls_or_fn
    self._handle = cls_or_fn if inspect.isfunction(cls_or_fn) else getattr(
        cls_or_fn, '__init__')
    input_is_python_file = module_path and module_path.endswith('.py')
    self._input_path = module_path if input_is_python_file else None
    self._required_fn_name_for_signature_parsing = getattr(cls_or_fn,
        'required_fn_name_for_signature_parsing', None)
