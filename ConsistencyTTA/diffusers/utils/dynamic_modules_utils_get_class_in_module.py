def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    module_path = module_path.replace(os.path.sep, '.')
    module = importlib.import_module(module_path)
    if class_name is None:
        return find_pipeline_class(module)
    return getattr(module, class_name)
