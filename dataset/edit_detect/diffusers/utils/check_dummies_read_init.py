def read_init():
    """Read the init and extracts PyTorch, TensorFlow, SentencePiece and Tokenizers objects."""
    with open(os.path.join(PATH_TO_DIFFUSERS, '__init__.py'), 'r', encoding
        ='utf-8', newline='\n') as f:
        lines = f.readlines()
    line_index = 0
    while not lines[line_index].startswith('if TYPE_CHECKING'):
        line_index += 1
    backend_specific_objects = {}
    while line_index < len(lines):
        backend = find_backend(lines[line_index])
        if backend is not None:
            while not lines[line_index].startswith('    else:'):
                line_index += 1
            line_index += 1
            objects = []
            while len(lines[line_index]) <= 1 or lines[line_index].startswith(
                ' ' * 8):
                line = lines[line_index]
                single_line_import_search = _re_single_line_import.search(line)
                if single_line_import_search is not None:
                    objects.extend(single_line_import_search.groups()[0].
                        split(', '))
                elif line.startswith(' ' * 12):
                    objects.append(line[12:-2])
                line_index += 1
            if len(objects) > 0:
                backend_specific_objects[backend] = objects
        else:
            line_index += 1
    return backend_specific_objects
