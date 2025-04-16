def _replace(match):
    imports = match.groups()[0]
    if ',' not in imports:
        return f'[{imports}]'
    keys = [part.strip().replace('"', '') for part in imports.split(',')]
    if len(keys[-1]) == 0:
        keys = keys[:-1]
    return '[' + ', '.join([f'"{k}"' for k in sort_objects(keys)]) + ']'
