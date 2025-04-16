def get_relative_imports(module_file):
    """
    Get the list of modules that are relatively imported in a module file.

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.
    """
    with open(module_file, 'r', encoding='utf-8') as f:
        content = f.read()
    relative_imports = re.findall('^\\s*import\\s+\\.(\\S+)\\s*$', content,
        flags=re.MULTILINE)
    relative_imports += re.findall('^\\s*from\\s+\\.(\\S+)\\s+import',
        content, flags=re.MULTILINE)
    return list(set(relative_imports))
