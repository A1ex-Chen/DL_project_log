def parse_requirements(file_path=ROOT.parent / 'requirements.txt', package=''):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (Path): Path to the requirements.txt file.
        package (str, optional): Python package to use instead of requirements.txt file, i.e. package='ultralytics'.

    Returns:
        (List[Dict[str, str]]): List of parsed requirements as dictionaries with `name` and `specifier` keys.

    Example:
        ```python
        from ultralytics.utils.checks import parse_requirements

        parse_requirements(package='ultralytics')
        ```
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if 
            'extra == ' not in x]
    else:
        requires = Path(file_path).read_text().splitlines()
    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith('#'):
            line = line.split('#')[0].strip()
            match = re.match('([a-zA-Z0-9-_]+)\\s*([<>!=~]+.*)?', line)
            if match:
                requirements.append(SimpleNamespace(name=match[1],
                    specifier=match[2].strip() if match[2] else ''))
    return requirements
