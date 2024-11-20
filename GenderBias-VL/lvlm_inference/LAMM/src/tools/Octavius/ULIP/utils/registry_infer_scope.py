@staticmethod
def infer_scope():
    """Infer the scope of registry.
        The name of the package where registry is defined will be returned.
        Example:
            # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.
        Returns:
            scope (str): The inferred scope name.
        """
    filename = inspect.getmodule(inspect.stack()[2][0]).__name__
    split_filename = filename.split('.')
    return split_filename[0]
