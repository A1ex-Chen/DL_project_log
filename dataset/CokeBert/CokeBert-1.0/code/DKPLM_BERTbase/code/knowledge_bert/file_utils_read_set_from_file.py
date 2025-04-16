def read_set_from_file(filename: str) ->Set[str]:
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, 'r', encoding='utf-8') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection
