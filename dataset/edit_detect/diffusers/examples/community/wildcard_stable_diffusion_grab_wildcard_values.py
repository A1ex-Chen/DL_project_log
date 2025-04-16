def grab_wildcard_values(wildcard_option_dict: Dict[str, List[str]]={},
    wildcard_files: List[str]=[]):
    for wildcard_file in wildcard_files:
        filename = get_filename(wildcard_file)
        read_values = read_wildcard_values(wildcard_file)
        if filename not in wildcard_option_dict:
            wildcard_option_dict[filename] = []
        wildcard_option_dict[filename].extend(read_values)
    return wildcard_option_dict
