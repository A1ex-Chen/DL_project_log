def replace(original_file: str, line_to_copy_below: str, lines_to_copy:
    List[str]):
    fh, abs_path = mkstemp()
    line_found = False
    with fdopen(fh, 'w') as new_file:
        with open(original_file) as old_file:
            for line in old_file:
                new_file.write(line)
                if line_to_copy_below in line:
                    line_found = True
                    for line_to_copy in lines_to_copy:
                        new_file.write(line_to_copy)
    if not line_found:
        raise ValueError(f'Line {line_to_copy_below} was not found in file.')
    copymode(original_file, abs_path)
    remove(original_file)
    move(abs_path, original_file)
