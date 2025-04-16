def new_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 and globals is not None and (globals.get('__package__',
        '') or '').startswith(_CFG_PACKAGE_NAME):
        cur_file = find_relative_file(globals['__file__'], name, level)
        _validate_py_syntax(cur_file)
        spec = importlib.machinery.ModuleSpec(_random_package_name(cur_file
            ), None, origin=cur_file)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = cur_file
        with PathManager.open(cur_file) as f:
            content = f.read()
        exec(compile(content, cur_file, 'exec'), module.__dict__)
        for name in fromlist:
            val = _cast_to_config(module.__dict__[name])
            module.__dict__[name] = val
        return module
    return old_import(name, globals, locals, fromlist=fromlist, level=level)
