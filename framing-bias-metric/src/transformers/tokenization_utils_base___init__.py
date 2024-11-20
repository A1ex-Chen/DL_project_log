def __init__(self, **kwargs):
    self.init_inputs = ()
    self.init_kwargs = copy.deepcopy(kwargs)
    self.name_or_path = kwargs.pop('name_or_path', '')
    model_max_length = kwargs.pop('model_max_length', kwargs.pop('max_len',
        None))
    self.model_max_length = (model_max_length if model_max_length is not
        None else VERY_LARGE_INTEGER)
    self.padding_side = kwargs.pop('padding_side', self.padding_side)
    assert self.padding_side in ['right', 'left'
        ], f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
    self.model_input_names = kwargs.pop('model_input_names', self.
        model_input_names)
    self.deprecation_warnings = {}
    super().__init__(**kwargs)
