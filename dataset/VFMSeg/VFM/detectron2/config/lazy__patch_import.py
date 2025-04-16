@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through PathManager
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip('.')
        for part in cur_name.split('.'):
            cur_file = os.path.join(cur_file, part)
        if not cur_file.endswith('.py'):
            cur_file += '.py'
        if not PathManager.isfile(cur_file):
            raise ImportError(
                f'Cannot import name {relative_import_path} from {original_file}: {cur_file} has to exist.'
                )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 and globals is not None and (globals.get(
            '__package__', '') or '').startswith(_CFG_PACKAGE_NAME):
            cur_file = find_relative_file(globals['__file__'], name, level)
            _validate_py_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(_random_package_name(
                cur_file), None, origin=cur_file)
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with PathManager.open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, 'exec'), module.__dict__)
            for name in fromlist:
                val = _cast_to_config(module.__dict__[name])
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level
            )
    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import
