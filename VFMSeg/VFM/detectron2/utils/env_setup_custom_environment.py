def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith('.py'):
        module = _import_file('detectron2.utils.env.custom_module',
            custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, 'setup_environment') and callable(module.
        setup_environment
        ), "Custom environment module defined in {} does not have the required callable attribute 'setup_environment'.".format(
        custom_module)
    module.setup_environment()
