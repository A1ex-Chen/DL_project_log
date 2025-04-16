def is_copy_consistent(filename, overwrite=False):
    """
    Check if the code commented as a copy in `filename` matches the original.
    Return the differences or overwrites the content depending on `overwrite`.
    """
    with open(filename, 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    diffs = []
    line_index = 0
    while line_index < len(lines):
        search = _re_copy_warning.search(lines[line_index])
        if search is None:
            line_index += 1
            continue
        indent, object_name, replace_pattern = search.groups()
        theoretical_code = find_code_in_diffusers(object_name)
        theoretical_indent = get_indent(theoretical_code)
        start_index = (line_index + 1 if indent == theoretical_indent else 
            line_index + 2)
        indent = theoretical_indent
        line_index = start_index
        should_continue = True
        while line_index < len(lines) and should_continue:
            line_index += 1
            if line_index >= len(lines):
                break
            line = lines[line_index]
            should_continue = _should_continue(line, indent) and re.search(
                f'^{indent}# End copy', line) is None
        while len(lines[line_index - 1]) <= 1:
            line_index -= 1
        observed_code_lines = lines[start_index:line_index]
        observed_code = ''.join(observed_code_lines)
        theoretical_code = [line for line in theoretical_code.split('\n') if
            _re_copy_warning.search(line) is None]
        theoretical_code = '\n'.join(theoretical_code)
        if len(replace_pattern) > 0:
            patterns = replace_pattern.replace('with', '').split(',')
            patterns = [_re_replace_pattern.search(p) for p in patterns]
            for pattern in patterns:
                if pattern is None:
                    continue
                obj1, obj2, option = pattern.groups()
                theoretical_code = re.sub(obj1, obj2, theoretical_code)
                if option.strip() == 'all-casing':
                    theoretical_code = re.sub(obj1.lower(), obj2.lower(),
                        theoretical_code)
                    theoretical_code = re.sub(obj1.upper(), obj2.upper(),
                        theoretical_code)
            theoretical_code = stylify(lines[start_index - 1] +
                theoretical_code)
            theoretical_code = theoretical_code[len(lines[start_index - 1]):]
        if observed_code != theoretical_code:
            diffs.append([object_name, start_index])
            if overwrite:
                lines = lines[:start_index] + [theoretical_code] + lines[
                    line_index:]
                line_index = start_index + 1
    if overwrite and len(diffs) > 0:
        print(f'Detected changes, rewriting {filename}.')
        with open(filename, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(lines)
    return diffs
