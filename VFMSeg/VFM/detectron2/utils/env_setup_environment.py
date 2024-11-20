def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $DETECTRON2_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True
    _configure_libraries()
    custom_module_path = os.environ.get('DETECTRON2_ENV_MODULE')
    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        pass
