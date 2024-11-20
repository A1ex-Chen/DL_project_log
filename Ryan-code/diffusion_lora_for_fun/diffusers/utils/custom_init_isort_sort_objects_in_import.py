def sort_objects_in_import(import_statement: str) ->str:
    """
    Sorts the imports in a single import statement.

    Args:
        import_statement (`str`): The import statement in which to sort the imports.

    Returns:
        `str`: The same as the input, but with objects properly sorted.
    """

    def _replace(match):
        imports = match.groups()[0]
        if ',' not in imports:
            return f'[{imports}]'
        keys = [part.strip().replace('"', '') for part in imports.split(',')]
        if len(keys[-1]) == 0:
            keys = keys[:-1]
        return '[' + ', '.join([f'"{k}"' for k in sort_objects(keys)]) + ']'
    lines = import_statement.split('\n')
    if len(lines) > 3:
        idx = 2 if lines[1].strip() == '[' else 1
        keys_to_sort = [(i, _re_strip_line.search(line).groups()[0]) for i,
            line in enumerate(lines[idx:-idx])]
        sorted_indices = sort_objects(keys_to_sort, key=lambda x: x[1])
        sorted_lines = [lines[x[0] + idx] for x in sorted_indices]
        return '\n'.join(lines[:idx] + sorted_lines + lines[-idx:])
    elif len(lines) == 3:
        if _re_bracket_content.search(lines[1]) is not None:
            lines[1] = _re_bracket_content.sub(_replace, lines[1])
        else:
            keys = [part.strip().replace('"', '') for part in lines[1].
                split(',')]
            if len(keys[-1]) == 0:
                keys = keys[:-1]
            lines[1] = get_indent(lines[1]) + ', '.join([f'"{k}"' for k in
                sort_objects(keys)])
        return '\n'.join(lines)
    else:
        import_statement = _re_bracket_content.sub(_replace, import_statement)
        return import_statement
