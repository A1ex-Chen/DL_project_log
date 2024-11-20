def __init__(self, *args, **kwargs):
    raise EnvironmentError(
        f'{self.__class__.__name__} is designed to be instantiated using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or `{self.__class__.__name__}.from_pipe(pipeline)` methods.'
        )
