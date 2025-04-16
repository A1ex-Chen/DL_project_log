def import_file(path, name: str=None, add_to_sys=True, disable_warning=False):
    global CUSTOM_LOADED_MODULES
    path = Path(path)
    module_name = path.stem
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    possible_paths = _get_possible_module_path(user_paths)
    model_import_name = _get_regular_import_name(path, possible_paths)
    if model_import_name is not None:
        return import_name(model_import_name)
    if name is not None:
        module_name = name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not disable_warning:
        logger.warning(
            f"Failed to perform regular import for file {path}. this means this file isn't in any folder in PYTHONPATH or don't have __init__.py in that project. directly file import may fail and some reflecting features are disabled even if import succeed. please add your project to PYTHONPATH or add __init__.py to ensure this file can be regularly imported. "
            )
    if add_to_sys:
        if (module_name in sys.modules and module_name not in
            CUSTOM_LOADED_MODULES):
            raise ValueError(f'{module_name} exists in system.')
        CUSTOM_LOADED_MODULES[module_name] = module
        sys.modules[module_name] = module
    return module
