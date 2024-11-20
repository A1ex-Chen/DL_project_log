def update_version_in_file(fname, version, pattern):
    """Update the version in one file using a specific pattern."""
    with open(fname, 'r', encoding='utf-8', newline='\n') as f:
        code = f.read()
    re_pattern, replace = REPLACE_PATTERNS[pattern]
    replace = replace.replace('VERSION', version)
    code = re_pattern.sub(replace, code)
    with open(fname, 'w', encoding='utf-8', newline='\n') as f:
        f.write(code)
