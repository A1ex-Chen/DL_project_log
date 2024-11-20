def check_submodules():
    spec = importlib.util.spec_from_file_location('transformers', os.path.
        join(PATH_TO_TRANSFORMERS, '__init__.py'),
        submodule_search_locations=[PATH_TO_TRANSFORMERS])
    transformers = spec.loader.load_module()
    module_not_registered = [module for module in
        get_transformers_submodules() if module not in IGNORE_SUBMODULES and
        module not in transformers._import_structure.keys()]
    if len(module_not_registered) > 0:
        list_of_modules = '\n'.join(f'- {module}' for module in
            module_not_registered)
        raise ValueError(
            f"""The following submodules are not properly registered in the main init of Transformers:
{list_of_modules}
Make sure they appear somewhere in the keys of `_import_structure` with an empty list as value."""
            )
