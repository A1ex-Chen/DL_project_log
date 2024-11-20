def sort_imports_in_all_inits(check_only=True):
    """
    Sort the imports defined in the `_import_structure` of all inits in the repo.

    Args:
        check_only (`bool`, *optional*, defaults to `True`): Whether or not to just check (and not auto-fix) the init.
    """
    failures = []
    for root, _, files in os.walk(PATH_TO_TRANSFORMERS):
        if '__init__.py' in files:
            result = sort_imports(os.path.join(root, '__init__.py'),
                check_only=check_only)
            if result:
                failures = [os.path.join(root, '__init__.py')]
    if len(failures) > 0:
        raise ValueError(
            f'Would overwrite {len(failures)} files, run `make style`.')
