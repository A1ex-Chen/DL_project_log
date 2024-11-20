def sort_imports(file: str, check_only: bool=True):
    """
    Sort the imports defined in the `_import_structure` of a given init.

    Args:
        file (`str`): The path to the init to check/fix.
        check_only (`bool`, *optional*, defaults to `True`): Whether or not to just check (and not auto-fix) the init.
    """
    with open(file, encoding='utf-8') as f:
        code = f.read()
    if '_import_structure' not in code:
        return
    main_blocks = split_code_in_indented_blocks(code, start_prompt=
        '_import_structure = {', end_prompt='if TYPE_CHECKING:')
    for block_idx in range(1, len(main_blocks) - 1):
        block = main_blocks[block_idx]
        block_lines = block.split('\n')
        line_idx = 0
        while line_idx < len(block_lines
            ) and '_import_structure' not in block_lines[line_idx]:
            if 'import dummy' in block_lines[line_idx]:
                line_idx = len(block_lines)
            else:
                line_idx += 1
        if line_idx >= len(block_lines):
            continue
        internal_block_code = '\n'.join(block_lines[line_idx:-1])
        indent = get_indent(block_lines[1])
        internal_blocks = split_code_in_indented_blocks(internal_block_code,
            indent_level=indent)
        pattern = _re_direct_key if '_import_structure = {' in block_lines[0
            ] else _re_indirect_key
        keys = [(pattern.search(b).groups()[0] if pattern.search(b) is not
            None else None) for b in internal_blocks]
        keys_to_sort = [(i, key) for i, key in enumerate(keys) if key is not
            None]
        sorted_indices = [x[0] for x in sorted(keys_to_sort, key=lambda x:
            x[1])]
        count = 0
        reordered_blocks = []
        for i in range(len(internal_blocks)):
            if keys[i] is None:
                reordered_blocks.append(internal_blocks[i])
            else:
                block = sort_objects_in_import(internal_blocks[
                    sorted_indices[count]])
                reordered_blocks.append(block)
                count += 1
        main_blocks[block_idx] = '\n'.join(block_lines[:line_idx] +
            reordered_blocks + [block_lines[-1]])
    if code != '\n'.join(main_blocks):
        if check_only:
            return True
        else:
            print(f'Overwriting {file}.')
            with open(file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(main_blocks))
