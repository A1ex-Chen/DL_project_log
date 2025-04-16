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
