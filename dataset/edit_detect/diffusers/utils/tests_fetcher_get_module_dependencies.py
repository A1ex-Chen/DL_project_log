def get_module_dependencies(module_fname: str, cache: Dict[str, List[str]]=None
    ) ->List[str]:
    """
    Refines the result of `extract_imports` to remove subfolders and get a proper list of module filenames: if a file
    as an import `from utils import Foo, Bar`, with `utils` being a subfolder containing many files, this will traverse
    the `utils` init file to check where those dependencies come from: for instance the files utils/foo.py and utils/bar.py.

    Warning: This presupposes that all intermediate inits are properly built (with imports from the respective
    submodules) and work better if objects are defined in submodules and not the intermediate init (otherwise the
    intermediate init is added, and inits usually have a lot of dependencies).

    Args:
        module_fname (`str`):
            The name of the file of the module where we want to look at the imports (given relative to the root of
            the repo).
        cache (Dictionary `str` to `List[str]`, *optional*):
            To speed up this function if it was previously called on `module_fname`, the cache of all previously
            computed results.

    Returns:
        `List[str]`: The list of module filenames imported in the input `module_fname` (with submodule imports refined).
    """
    dependencies = []
    imported_modules = extract_imports(module_fname, cache=cache)
    while len(imported_modules) > 0:
        new_modules = []
        for module, imports in imported_modules:
            if module.endswith('__init__.py'):
                new_imported_modules = extract_imports(module, cache=cache)
                for new_module, new_imports in new_imported_modules:
                    if any(i in new_imports for i in imports):
                        if new_module not in dependencies:
                            new_modules.append((new_module, [i for i in
                                new_imports if i in imports]))
                        imports = [i for i in imports if i not in new_imports]
                if len(imports) > 0:
                    path_to_module = PATH_TO_REPO / module.replace(
                        '__init__.py', '')
                    dependencies.extend([os.path.join(module.replace(
                        '__init__.py', ''), f'{i}.py') for i in imports if
                        (path_to_module / f'{i}.py').is_file()])
                    imports = [i for i in imports if not (path_to_module /
                        f'{i}.py').is_file()]
                    if len(imports) > 0:
                        dependencies.append(module)
            else:
                dependencies.append(module)
        imported_modules = new_modules
    return dependencies
