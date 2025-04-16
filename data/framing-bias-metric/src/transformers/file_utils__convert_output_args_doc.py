def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ''
    for line in output_args_doc.split('\n'):
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f'{line}\n'
        else:
            current_block += f'{line[2:]}\n'
    blocks.append(current_block[:-1])
    for i in range(len(blocks)):
        blocks[i] = re.sub('^(\\s+)(\\S+)(\\s+)', '\\1- **\\2**\\3', blocks[i])
        blocks[i] = re.sub(':\\s*\\n\\s*(\\S)', ' -- \\1', blocks[i])
    return '\n'.join(blocks)
