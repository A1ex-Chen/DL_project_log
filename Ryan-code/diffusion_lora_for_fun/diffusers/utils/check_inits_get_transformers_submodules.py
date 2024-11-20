def get_transformers_submodules():
    """
    Returns the list of Transformers submodules.
    """
    submodules = []
    for path, directories, files in os.walk(PATH_TO_TRANSFORMERS):
        for folder in directories:
            if folder.startswith('_'):
                directories.remove(folder)
                continue
            if len(list((Path(path) / folder).glob('*.py'))) == 0:
                continue
            short_path = str((Path(path) / folder).relative_to(
                PATH_TO_TRANSFORMERS))
            submodule = short_path.replace(os.path.sep, '.')
            submodules.append(submodule)
        for fname in files:
            if fname == '__init__.py':
                continue
            short_path = str((Path(path) / fname).relative_to(
                PATH_TO_TRANSFORMERS))
            submodule = short_path.replace('.py', '').replace(os.path.sep, '.')
            if len(submodule.split('.')) == 1:
                submodules.append(submodule)
    return submodules
