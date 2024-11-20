def extract_imports(module_fname: str, cache: Dict[str, List[str]]=None
    ) ->List[str]:
    """
    Get the imports a given module makes.

    Args:
        module_fname (`str`):
            The name of the file of the module where we want to look at the imports (given relative to the root of
            the repo).
        cache (Dictionary `str` to `List[str]`, *optional*):
            To speed up this function if it was previously called on `module_fname`, the cache of all previously
            computed results.

    Returns:
        `List[str]`: The list of module filenames imported in the input `module_fname` (a submodule we import from that
        is a subfolder will give its init file).
    """
    if cache is not None and module_fname in cache:
        return cache[module_fname]
    with open(PATH_TO_REPO / module_fname, 'r', encoding='utf-8') as f:
        content = f.read()
    splits = content.split('"""')
    content = ''.join(splits[::2])
    module_parts = str(module_fname).split(os.path.sep)
    imported_modules = []
    relative_imports = _re_single_line_relative_imports.findall(content)
    relative_imports = [(mod, imp) for mod, imp in relative_imports if 
        '# tests_ignore' not in imp and imp.strip() != '(']
    multiline_relative_imports = _re_multi_line_relative_imports.findall(
        content)
    relative_imports += [(mod, imp) for mod, imp in
        multiline_relative_imports if '# tests_ignore' not in imp]
    for module, imports in relative_imports:
        level = 0
        while module.startswith('.'):
            module = module[1:]
            level += 1
        if len(module) > 0:
            dep_parts = module_parts[:len(module_parts) - level
                ] + module.split('.')
        else:
            dep_parts = module_parts[:len(module_parts) - level]
        imported_module = os.path.sep.join(dep_parts)
        imported_modules.append((imported_module, [imp.strip() for imp in
            imports.split(',')]))
    direct_imports = _re_single_line_direct_imports.findall(content)
    direct_imports = [(mod, imp) for mod, imp in direct_imports if 
        '# tests_ignore' not in imp and imp.strip() != '(']
    multiline_direct_imports = _re_multi_line_direct_imports.findall(content)
    direct_imports += [(mod, imp) for mod, imp in multiline_direct_imports if
        '# tests_ignore' not in imp]
    for module, imports in direct_imports:
        import_parts = module.split('.')[1:]
        dep_parts = ['src', 'diffusers'] + import_parts
        imported_module = os.path.sep.join(dep_parts)
        imported_modules.append((imported_module, [imp.strip() for imp in
            imports.split(',')]))
    result = []
    for module_file, imports in imported_modules:
        if (PATH_TO_REPO / f'{module_file}.py').is_file():
            module_file = f'{module_file}.py'
        elif (PATH_TO_REPO / module_file).is_dir() and (PATH_TO_REPO /
            module_file / '__init__.py').is_file():
            module_file = os.path.sep.join([module_file, '__init__.py'])
        imports = [imp for imp in imports if len(imp) > 0 and re.match(
            '^[A-Za-z0-9_]*$', imp)]
        if len(imports) > 0:
            result.append((module_file, imports))
    if cache is not None:
        cache[module_fname] = result
    return result
