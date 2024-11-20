def get_indent(code):
    lines = code.split('\n')
    idx = 0
    while idx < len(lines) and len(lines[idx]) == 0:
        idx += 1
    if idx < len(lines):
        return re.search('^(\\s*)\\S', lines[idx]).groups()[0]
    return ''
