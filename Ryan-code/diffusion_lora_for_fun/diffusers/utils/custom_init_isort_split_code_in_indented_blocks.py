def split_code_in_indented_blocks(code: str, indent_level: str='',
    start_prompt: Optional[str]=None, end_prompt: Optional[str]=None) ->List[
    str]:
    """
    Split some code into its indented blocks, starting at a given level.

    Args:
        code (`str`): The code to split.
        indent_level (`str`): The indent level (as string) to use for identifying the blocks to split.
        start_prompt (`str`, *optional*): If provided, only starts splitting at the line where this text is.
        end_prompt (`str`, *optional*): If provided, stops splitting at a line where this text is.

    Warning:
        The text before `start_prompt` or after `end_prompt` (if provided) is not ignored, just not split. The input `code`
        can thus be retrieved by joining the result.

    Returns:
        `List[str]`: The list of blocks.
    """
    index = 0
    lines = code.split('\n')
    if start_prompt is not None:
        while not lines[index].startswith(start_prompt):
            index += 1
        blocks = ['\n'.join(lines[:index])]
    else:
        blocks = []
    current_block = [lines[index]]
    index += 1
    while index < len(lines) and (end_prompt is None or not lines[index].
        startswith(end_prompt)):
        if len(lines[index]) > 0 and get_indent(lines[index]) == indent_level:
            if len(current_block) > 0 and get_indent(current_block[-1]
                ).startswith(indent_level + ' '):
                current_block.append(lines[index])
                blocks.append('\n'.join(current_block))
                if index < len(lines) - 1:
                    current_block = [lines[index + 1]]
                    index += 1
                else:
                    current_block = []
            else:
                blocks.append('\n'.join(current_block))
                current_block = [lines[index]]
        else:
            current_block.append(lines[index])
        index += 1
    if len(current_block) > 0:
        blocks.append('\n'.join(current_block))
    if end_prompt is not None and index < len(lines):
        blocks.append('\n'.join(lines[index:]))
    return blocks
