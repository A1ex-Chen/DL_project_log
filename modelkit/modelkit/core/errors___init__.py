def __init__(self, *args, **kwargs):
    super().__init__(*args, error_str=
        'Predict item validation error when calling model', pydantic_exc=
        kwargs.pop('pydantic_exc'))
