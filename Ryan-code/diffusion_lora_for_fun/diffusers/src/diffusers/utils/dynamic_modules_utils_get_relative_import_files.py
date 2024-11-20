def get_relative_import_files(module_file):
    """
    Get the list of all files that are needed for a given module. Note that this function recurses through the relative
    imports (if a imports b and b imports c, it will return module files for b and c).

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.
    """
    no_change = False
    files_to_check = [module_file]
    all_relative_imports = []
    while not no_change:
        new_imports = []
        for f in files_to_check:
            new_imports.extend(get_relative_imports(f))
        module_path = Path(module_file).parent
        new_import_files = [str(module_path / m) for m in new_imports]
        new_import_files = [f for f in new_import_files if f not in
            all_relative_imports]
        files_to_check = [f'{f}.py' for f in new_import_files]
        no_change = len(new_import_files) == 0
        all_relative_imports.extend(files_to_check)
    return all_relative_imports
