def check_imports(filename):
    """
    Check if the current Python environment contains all the libraries that are imported in a file.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    imports = re.findall('^\\s*import\\s+(\\S+)\\s*$', content, flags=re.
        MULTILINE)
    imports += re.findall('^\\s*from\\s+(\\S+)\\s+import', content, flags=
        re.MULTILINE)
    imports = [imp.split('.')[0] for imp in imports if not imp.startswith('.')]
    imports = list(set(imports))
    missing_packages = []
    for imp in imports:
        try:
            importlib.import_module(imp)
        except ImportError:
            missing_packages.append(imp)
    if len(missing_packages) > 0:
        raise ImportError(
            f"This modeling file requires the following packages that were not found in your environment: {', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
            )
    return get_relative_imports(filename)
