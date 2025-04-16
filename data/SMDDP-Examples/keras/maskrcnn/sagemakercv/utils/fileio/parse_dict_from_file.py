def dict_from_file(filename, key_type=str):
    """Load a text file and parse the content as a dict.

    Each line of the text file will be two or more columns splited by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict's keys. str is user by default and
            type conversion will be performed if specified.

    Returns:
        dict: The parsed contents.
    """
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
