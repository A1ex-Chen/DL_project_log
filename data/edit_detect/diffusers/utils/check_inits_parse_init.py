def parse_init(init_file):
    """
    Read an init_file and parse (per backend) the _import_structure objects defined and the TYPE_CHECKING objects
    defined
    """
    with open(init_file, 'r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()
    line_index = 0
    while line_index < len(lines) and not lines[line_index].startswith(
        '_import_structure = {'):
        line_index += 1
    if line_index >= len(lines):
        return None
    objects = []
    while not lines[line_index].startswith('if TYPE_CHECKING'
        ) and find_backend(lines[line_index]) is None:
        line = lines[line_index]
        if _re_one_line_import_struct.search(line):
            content = _re_one_line_import_struct.search(line).groups()[0]
            imports = re.findall('\\[([^\\]]+)\\]', content)
            for imp in imports:
                objects.extend([obj[1:-1] for obj in imp.split(', ')])
            line_index += 1
            continue
        single_line_import_search = _re_import_struct_key_value.search(line)
        if single_line_import_search is not None:
            imports = [obj[1:-1] for obj in single_line_import_search.
                groups()[0].split(', ') if len(obj) > 0]
            objects.extend(imports)
        elif line.startswith(' ' * 8 + '"'):
            objects.append(line[9:-3])
        line_index += 1
    import_dict_objects = {'none': objects}
    while not lines[line_index].startswith('if TYPE_CHECKING'):
        backend = find_backend(lines[line_index])
        if _re_try.search(lines[line_index - 1]) is None:
            backend = None
        if backend is not None:
            line_index += 1
            while _re_else.search(lines[line_index]) is None:
                line_index += 1
            line_index += 1
            objects = []
            while len(lines[line_index]) <= 1 or lines[line_index].startswith(
                ' ' * 4):
                line = lines[line_index]
                if _re_import_struct_add_one.search(line) is not None:
                    objects.append(_re_import_struct_add_one.search(line).
                        groups()[0])
                elif _re_import_struct_add_many.search(line) is not None:
                    imports = _re_import_struct_add_many.search(line).groups()[
                        0].split(', ')
                    imports = [obj[1:-1] for obj in imports if len(obj) > 0]
                    objects.extend(imports)
                elif _re_between_brackets.search(line) is not None:
                    imports = _re_between_brackets.search(line).groups()[0
                        ].split(', ')
                    imports = [obj[1:-1] for obj in imports if len(obj) > 0]
                    objects.extend(imports)
                elif _re_quote_object.search(line) is not None:
                    objects.append(_re_quote_object.search(line).groups()[0])
                elif line.startswith(' ' * 8 + '"'):
                    objects.append(line[9:-3])
                elif line.startswith(' ' * 12 + '"'):
                    objects.append(line[13:-3])
                line_index += 1
            import_dict_objects[backend] = objects
        else:
            line_index += 1
    objects = []
    while line_index < len(lines) and find_backend(lines[line_index]
        ) is None and not lines[line_index].startswith('else'):
        line = lines[line_index]
        single_line_import_search = _re_import.search(line)
        if single_line_import_search is not None:
            objects.extend(single_line_import_search.groups()[0].split(', '))
        elif line.startswith(' ' * 8):
            objects.append(line[8:-2])
        line_index += 1
    type_hint_objects = {'none': objects}
    while line_index < len(lines):
        backend = find_backend(lines[line_index])
        if _re_try.search(lines[line_index - 1]) is None:
            backend = None
        if backend is not None:
            line_index += 1
            while _re_else.search(lines[line_index]) is None:
                line_index += 1
            line_index += 1
            objects = []
            while len(lines[line_index]) <= 1 or lines[line_index].startswith(
                ' ' * 8):
                line = lines[line_index]
                single_line_import_search = _re_import.search(line)
                if single_line_import_search is not None:
                    objects.extend(single_line_import_search.groups()[0].
                        split(', '))
                elif line.startswith(' ' * 12):
                    objects.append(line[12:-2])
                line_index += 1
            type_hint_objects[backend] = objects
        else:
            line_index += 1
    return import_dict_objects, type_hint_objects
